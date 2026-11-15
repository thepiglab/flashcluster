# flashcluster

Fast k-means clustering and exact or IVF nearest-neighbour search for embedding vectors, written in Rust, with
explicit SIMD on CPUs (AVX2, AVX-512, VNNI) and CUDA kernels on NVIDIA GPUs (cuBLAS, CUTLASS).

- **e-means.** Mini-batch k-means with an exponential moving average and fitted batch-size and step-size laws, from
  "E-Means: Scaling Laws for Mini-Batch K-Means" (arXiv 2611.09501).
  - Ported from FAISS's MIT-licensed reference (`faiss/contrib/e_means`), with the same constants, schedule, revival
    and closing means.
  - FAISS-style Lloyd k-means is included as the comparison arm.
- **CPU search.** IVF over f16 or int8 lists, list-major batch search, in-place build and quantisation (one copy of
  the corpus in memory), and an optional exact re-rank from the memory-mapped f16 file.
- **GPU search.** Exact top-k on tensor cores.
  - Precisions: f16 with f32 accumulation, or int8 (CUTLASS tcgen05 on SM100, mma.sync on SM120, a dp4a kernel for
    small batches).
  - Device top-k: segment sorts, then a threshold filter.
  - Optional exact f16 re-rank of an int8 shortlist, from device memory (speed) or from pinned host memory (low
    memory).
- **GPU e-means.** Centroids and sums stay on the device. The assignment step runs in f32, TF32, f16, bf16, int8 or
  FP8; centroid updates are always f32.

## Results

**Dataset:** 2,433,196 text-chunk embeddings (768-d, L2-normalised, f16 source precision) and 1,962 real queries.
Ground truth is the exact inner-product top-100.

**Metrics:**
- **overlap@10:** the share of the exact top-10 that an engine returns.
- **R@10:** the query's target document within the top-10 distinct documents.

**Method:** see [Method](#method). Full tables, including precision rungs not recommended here, are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### GPU search: ms per call (queries per call: 1 / 64 / 256 / 1,962)

NVIDIA B200:

| Engine | overlap@10 | R@10 | Device memory | 1 | 64 | 256 | 1,962 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **flashcluster f16, exact** | 0.9999 | 0.9745 | 4.4 GB | 1.50 | 1.67 | 3.06 | 14.8 |
| **flashcluster int8 + exact re-rank 200, speed** | 0.9999 | 0.9745 | 6.2 GB | **1.02** | **1.41** | **1.56** | **12.1** |
| **flashcluster int8 + exact re-rank 200, low memory** | 0.9999 | 0.9745 | **3.0 GB** | 1.09 | 1.78 | 3.18 | 25.5 |
| flashcluster int8 (approximate) | 0.978 | 0.975 | **2.6 GB** | 0.95 | 1.33 | 1.60 | 12.4 |
| cuVS brute force, f16 | 0.9995 | 0.9745 | 4.7 GB | 2.14 | 2.35 | 5.49 | 29.1 |
| cuVS brute force, f32 | 0.9995 | 0.9745 | 11.8 GB | 1.64 | 5.73 | 19.2 | 146 |
| cuVS IVF-Flat f16, nprobe 128 | 0.899 | 0.965 | 10.5 GB | 2.11 | — | — | 76.0 |
| cuVS IVF-Flat f32, nprobe 128 | 0.950 | 0.968 | 23.2 GB | 2.34 | — | — | 108 |

NVIDIA RTX PRO 6000 Blackwell (queries per call: 1 / 64 / 256 / 512 / 1,962):

| Engine | overlap@10 | Device memory | 1 | 64 | 256 | 512 | 1,962 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **flashcluster f16, exact** | 0.9999 | 4.6 GB | 2.81 | 3.72 | 5.92 | 9.25 | 35.6 |
| **flashcluster int8 + exact re-rank 200, speed** | 0.9999 | 6.1 GB | **1.43** | **3.18** | **3.77** | **7.03** | **26.7** |
| **flashcluster int8 + exact re-rank 200, low memory** | 0.9999 | **2.6 GB** | 1.48 | 3.52 | 5.23 | 9.99 | 38.6 |
| flashcluster int8 (approximate) | 0.978 | 2.9 GB | 1.37 | 3.09 | 3.77 | 7.00 | 26.4 |
| cuVS brute force, f16 | 0.9995 | 4.7 GB | 3.36 | 5.04 | 8.59 | — | 42.3 |
| cuVS brute force, f32 | 0.9995 | 11.8 GB | 5.36 | 8.59 | 20.1 | — | 150 |
| cuVS IVF-Flat f16, nprobe 128 | 0.897 | 10.4 GB | 1.89 | — | — | — | 195 |
| cuVS IVF-Flat f32, nprobe 128 | 0.950 | 23.2 GB | 2.16 | — | — | — | 418 |

"—" means not measured. cuVS IVF was timed at 1 and 1,962 queries only.

### GPU k-means: k = 4096 on 2.43M × 768

| Engine | B200 | RTX PRO 6000 | Objective (lower is better) | Device memory |
|---|---:|---:|---:|---:|
| **flashcluster e-means, int8 assignment** | **0.42 s** | **0.57 s** | 0.244612 | 4.9 GB |
| flashcluster e-means, f16 assignment | 0.48 s | 0.64 s | 0.244651 | 4.9 GB |
| flashcluster e-means, TF32 | 0.78 s | 0.81 s | 0.244644 | 4.9 GB |
| flashcluster e-means, f32 | 1.79 s | 1.73 s | 0.244644 | 4.9 GB |
| cuML KMeans (25 iterations, k-means‖) | 25.7 s | 37.4 s | 0.245770 | 20-24 GB |
| cuVS kmeans (25 iterations) | 25.4 s | 33.4 s | 0.24572-0.24580 | 20-24 GB |

e-means uses the quality batch law and 5 epochs; centroid updates are always f32. The objective is the mean squared
distance to the nearest centroid, recomputed exactly for every engine.

### CPU search: IVF, k = 4096, 32 threads (latency on 1 thread, one query per call)

AMD EPYC 9655P (Zen 5, AVX-512 VNNI):

| Engine | overlap@10 at nprobe 128 | Build | Peak RSS | Latency ms, nprobe 8 / 32 / 128 | Queries/s, nprobe 8 / 32 / 128 |
|---|---:|---:|---:|---|---|
| **flashcluster f16** | 0.9738 | 3.3 s | 4.40 GB | 0.48 / 1.32 / 4.90 | 51.5k / 27.6k / 9.6k |
| **flashcluster int8** | 0.9577 | 3.5 s | **4.16 GB** | **0.28 / 0.69 / 2.03** | **77.7k / 30.1k / 9.6k** |
| **flashcluster int8 + exact re-rank 200** | 0.9738 | 3.5 s | 4.20 GB | 0.51 / 0.81 / 2.35 | 54.2k / 19.8k / 6.4k |
| FAISS IVF-Flat | 0.9731 | 23.5 s | 13.51 GB | 0.57 / 1.74 / 6.31 | 11.4k / 3.6k / 0.83k |
| FAISS IVF-SQfp16 | 0.9731 | 24.5 s | 8.83 GB | 0.52 / 1.46 / 5.26 | 20.9k / 6.4k / 1.8k |
| FAISS IVF-SQ8 | 0.9677 | 24.1 s | 4.76 GB | 0.40 / 0.98 / 2.94 | 32.7k / 12.2k / 3.2k |
| FAISS IVF-SQ8 + IndexRefineFlat (re-rank 200) | 0.9731 | 28.2 s | 14.76 GB | 0.46 / 1.04 / 3.03 | 27.9k / 10.2k / 3.3k |

AMD EPYC 7702 (Zen 2, AVX2):

| Engine | overlap@10 at nprobe 128 | Build | Peak RSS | Latency ms, nprobe 8 / 32 / 128 | Queries/s, nprobe 8 / 32 / 128 |
|---|---:|---:|---:|---|---|
| **flashcluster f16** | 0.9738 | 8.4 s | 5.01 GB | 1.65 / 4.03 / 11.9 | 17.8k / 13.7k / 5.0k |
| flashcluster int8 † | 0.9577 | 8.5 s | 6.50 GB † | 1.08 / 3.12 / 7.61 | 33.3k / 12.6k / 4.1k |
| FAISS IVF-Flat | 0.9731 | 55.8 s | 13.50 GB | 2.15 / 6.69 / 20.9 | 5.1k / 1.1k / 0.32k |
| FAISS IVF-SQfp16 | 0.9731 | 59.2 s | 8.83 GB | 3.31 / 6.60 / 20.9 | 3.1k / 1.4k / 0.88k |
| FAISS IVF-SQ8 | 0.9677 | 46.8 s | 5.05 GB | 1.88 / 5.45 / 17.4 | 12.4k / 4.1k / 1.2k |

† The Zen 2 int8 run used an earlier build, from before in-place quantisation (one-copy build peak) and the 4-query
batch kernel. Its RSS and batch numbers are not current.

### CPU k-means: k = 4096 on 2.43M × 768, 32 threads

| Engine | Zen 5 (EPYC 9655P) | Zen 2 (EPYC 7702) | Objective |
|---|---:|---:|---:|
| **flashcluster e-means** (quality law, 5 epochs) | **25.8 s** | **59.9 s** | **0.24463** |
| flashcluster Lloyd (FAISS-style, 25 iterations, 256 points per centroid) | 43.7 s | 109 s | 0.24645 |


## How it is fast

1. **Fewer bytes.**
   - Corpus vectors are stored as f16 (exact for these embeddings) or int8, and memory-bound scans read half or a
     quarter of the bytes of f32.
   - Large-batch GPU scores are written as f16 (the int8 CUTLASS epilogue scales int32 to f16).
   - The int8 index is quantised in place inside the f16 allocation.
2. **Explicit SIMD.**
   - CPU: AVX2+F16C and AVX-512 kernels score 8 rows per call; VNNI `vpdpbusd` scores 4 queries per row load.
   - GPU: dp4a for 1-8 queries.
   - All are chosen at run time.
3. **List-major batch search (CPU).** Each IVF list is read once for every query that probes it, with an f32 GEMM tile
   for hot lists. FAISS scans per query.
4. **Device top-k without full sorts (GPU).**
   - The first chunk gets exact segment sorts.
   - After that, a per-query threshold filter keeps only candidates that can still enter the top-k.
   - Merges choose their sort size per row on the device, with no host round trips.
   - If a row ever overflows, that work is redone exactly.
5. **e-means instead of Lloyd.** EMA mini-batch updates reach a lower objective in about 5 epochs. On GPU the
   centroids never leave the device.
6. **Exact re-rank of a shortlist.** int8 finds the top 200 and f16 × f16 → f32 re-scores it, recovering the exact
   top-100 (recall@100 0.9998).

## Method

**Same inputs.** Every engine reads one `flashcluster prep` directory: f16 corpus, f32 queries, shared k = 4096
centroids, and the exact ground truth.

**Isolation and timing.**
- Each engine runs in its own process; peak device memory is sampled with nvidia-smi and host memory is peak RSS.
- GPU timings follow one untimed call at each batch size.
- CPU latency is one query per call on one thread; throughput is the whole query batch on 32 threads, median of 3.
- Reference engines (FAISS 1.16.0, cuVS 26.10, cuML 26.10) ran on the same CPU or GPU model as flashcluster.
- CPU references ran in the same job and on the same node. GPU references ran in separate jobs on the same GPU model.

**Hardware.** NVIDIA B200 (183 GB) and RTX PRO 6000 Blackwell Server Edition (98 GB), driver 580; AMD EPYC 9655P and
EPYC 7702; CUDA 13.2; Rust 1.99.

## Build

```bash
cargo build --release                       # CPU (AVX2 / AVX-512 / VNNI chosen at run time)
cargo build --release --features cuda       # + GPU (needs nvcc; CUDA >= 12.8 for Blackwell)
cargo test --release
```

**CUTLASS int8 kernels (optional).** CUTLASS is not vendored. Fetch NVIDIA CUTLASS (BSD-3-Clause), headers only, and
point the build at it:

```bash
git clone --depth 1 --branch v4.7.1 https://github.com/NVIDIA/cutlass.git third_party/cutlass-4.7.1
FLASHCLUSTER_CUTLASS=$PWD/third_party/cutlass-4.7.1 cargo build --release --features cuda
```

- **SM targets:** `FLASHCLUSTER_CUDA_ARCH` picks them; the default is `100,120`.
- **Kernel dispatch:** the SM100 CUTLASS kernel is built for `sm_100a` and the SM120 kernel for `sm_120`. Other GPUs
  use cuBLAS.
- **Runtime switches:**
  - `FLASHCLUSTER_SIMD=avx2|scalar` and `FLASHCLUSTER_CUTLASS_GEMM=0` force a code path.
  - `FLASHCLUSTER_HALF_MIN_NQ` sets the batch size where the large-batch GPU path starts (default 512 on SM100, 128
    on SM120).

## Use

```bash
# cluster an .npy matrix (f16/f32/f64, n x d); rows are L2-normalised unless --no-normalize
flashcluster fit --data vectors.npy --k 4096 --out run/          # centroids.f32.npy, labels.u32.npy, fit.json

# shared benchmark inputs: f16 data, queries, centroids, exact top-100 (optional groups/targets for document recall)
flashcluster prep --data corpus.npy --queries queries.npy --k 4096 --out prep/
flashcluster search-bench --prep prep/ [--int8 [--rerank 200]] --out cpu.json
flashcluster gpu-bench --prep prep/ --precision f16|int8g|int8|fp8|bf16|f16acc [--rerank 200 [--rerank-device]] \
  [--emeans --emeans-precision f32|tf32|f16|bf16|int8|fp8] --out gpu.json
```

Library: `flashcluster::kmeans::{fit, emeans, emeans_with}`, `flashcluster::index::Ivf`, and `flashcluster::gpu::Gpu`
(feature `cuda`).

## License

MIT (see [LICENSE](LICENSE)).

- e-means follows the FAISS reference implementation (MIT, Meta Platforms) and the e-means paper (arXiv 2611.09501);
  please cite the paper if you use it.
- The optional CUTLASS kernels compile against NVIDIA CUTLASS (BSD-3-Clause), which is fetched separately, not
  vendored.
