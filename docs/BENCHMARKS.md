# Benchmarks

## Dataset and method

All engines read one shared `flashcluster prep` directory:

- **Corpus:** 2,433,196 text-chunk embeddings (EmbeddingGemma 2, 768-d, L2-normalised, stored as f16 because that is
  the source precision).
- **Queries:** 1,962 real queries.
- **Ground truth:** the exact inner-product top-100, computed in f32 math.
- **Group recall:** a group id per chunk (its source document).
- **Clustering:** k = 4096 shared IVF centroids.

How runs were measured:

- Each engine runs in its own process.
- Peak device memory is sampled by nvidia-smi; host memory is peak RSS.
- GPU timings follow one untimed call at each batch size.
- CPU and GPU references (FAISS 1.16.0, cuVS 26.10, cuML 26.10) ran on the same node type in the same job as
  flashcluster.

Metrics:

- **overlap@10 / recall@100:** share of the exact top-10 / top-100 returned.
- **R@10:** the query's target document within the top-10 distinct documents.

## GPU search (exact)

These f16 runs predate the int8 kernels. Current f16 and int8 numbers are in the v3 tables below and in the README;
the reference engines here are the ones those tables compare against.

flashcluster's GPU search is exact. It uses f16 tensor-core GEMM with f32 accumulation and a device top-k.

| Queries per call | B200: flashcluster | B200: cuVS bf f16 | B200: cuVS bf f32 | RTX PRO 6000: flashcluster | RTX PRO 6000: cuVS bf f16 | RTX PRO 6000: cuVS bf f32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **1.51 ms** | 2.14 | 1.64 | **2.81 ms** | 3.36 | 5.36 |
| 64 | **1.67 ms** | 2.35 | 5.73 | **3.72 ms** | 5.04 | 8.59 |
| 256 | **3.35 ms** | 5.49 | 19.2 | **7.15 ms** | 8.59 | 20.1 |
| 1,962 | **15.2 ms** | 29.1 | 146 | **35.7 ms** | 42.3 | 150 |
| Peak device memory | **4.27-4.53 GB** | 4.72 | 11.8 | **4.20-4.45 GB** | 4.65 | 11.8 |
| Peak host RSS | **4.2-4.5 GB** | 8.5 | 8.6 | **4.4 GB** | 8.5 | 8.5 |

R@10 is 0.9745 for every exact engine. overlap@10 against the f32 ground truth is 0.9999 for flashcluster and 0.9995
for cuVS.

**cuVS IVF-Flat** (approximate) tops out at R@10 0.965-0.971 at nprobe 128, which costs:

| GPU | 1 query | Batch | Device memory |
|---|---:|---:|---:|
| B200 | 2.1 ms | 76 ms | 10.5 GB (f16) / 23 GB (f32) |
| RTX PRO 6000 | 1.9 ms | 195 ms | 10.5 GB (f16) / 23 GB (f32) |

At equal or better recall, exact flashcluster search is faster at every batch size.

## Clustering: k = 4096 on 2.43M × 768

| Engine | B200 | RTX PRO 6000 | Objective (lower is better) | Device memory |
|---|---:|---:|---:|---:|
| **flashcluster e-means, f16 assignment** | **0.47 s** | **0.64 s** | 0.244651 | 4.9 GB |
| flashcluster e-means, TF32 assignment | 0.52 s | 0.81 s | 0.244644 | 4.9 GB |
| flashcluster e-means, f32 | 1.68 s | 1.73 s | 0.244644 | 4.9 GB |
| cuML KMeans (25 iterations, k-means‖) | 25.7 s | 37.4 s | 0.24577 | 20-24 GB |
| cuVS kmeans (25 iterations) | 25.4 s | 33.4 s | 0.24572-0.24580 | 20-24 GB |
| flashcluster e-means, CPU (32 × EPYC 7702) | 38-66 s | | 0.24463 | (host) |

## Precision ladder (GPU), v3: recommended rungs

**Gate:** overlap@10 within 0.001 of exact (≥ 0.9989). "Approximate" rungs miss it by design and are listed for
speed and memory.

**Recommended set.** Both GPUs use CUTLASS int8 kernels: tcgen05 on B200 (SM100), mma.sync on RTX PRO 6000 (SM120).

**B200** (ms at 1 / 64 / 256 / 1,962 queries):

| Rung | overlap@10 | Device memory | Time |
|---|---:|---:|---|
| f16 storage, f32 accumulate (exact) | 0.9999 | 4.4 GB | 1.50 / 1.67 / 3.06 / 14.8 |
| **int8 + exact re-rank 200, speed mode** (f16 copy on device) | **0.9999** | 6.2 GB | **1.02 / 1.41 / 1.56 / 12.1** |
| **int8 + exact re-rank 200, low-memory mode** (pinned host f16) | **0.9999** | **3.0 GB** | **1.09** / 1.78 / 3.18 / 25.5 |
| int8, global scale (approximate) | 0.978 | **2.6 GB** | **0.95 / 1.33 / 1.60 / 12.4** |

**RTX PRO 6000** (ms at 1 / 64 / 256 / 512 / 1,962 queries):

| Rung | overlap@10 | Device memory | Time |
|---|---:|---:|---|
| f16 storage, f32 accumulate (exact) | 0.9999 | 4.6 GB | 2.81 / 3.72 / 5.92 / 9.25 / 35.6 |
| **int8 + exact re-rank 200, speed mode** | **0.9999** | 6.1 GB | **1.43 / 3.18 / 3.77 / 7.03 / 26.7** |
| **int8 + exact re-rank 200, low-memory mode** | **0.9999** | **2.6 GB** | **1.48 / 3.52 / 5.23** / 9.99 / 38.6 |
| int8, global scale (approximate) | 0.978 | **2.9 GB** | **1.37 / 3.09 / 3.77 / 7.00 / 26.4** |

- **Speed mode and plain int8 are faster than f16 at every batch size on both GPUs.**
- **Low-memory mode** is the documented memory-for-speed choice. It is faster than f16 up to 256 queries and slower
  at larger batches, because the shortlist rows cross PCIe, but it uses about 35% less device memory.
- **Reference, cuVS brute force f16:** B200 2.14 / 2.35 / 5.49 / 29.1 ms; RTX 3.36 / 5.04 / 8.59 / 42.3 ms.

**Not recommended:**

- **bf16 and f16 accumulation.** On B200 and RTX PRO 6000 they move the same bytes and run the same tensor-core rate
  as f16, so they cannot be faster. Both fail the gate: bf16 overlap@10 is 0.993, because its 7-bit mantissa costs
  ranking ties; f16 accumulation is 0.979, from rounding over 768 terms.
- **FP8 e4m3.** Plain overlap@10 is 0.922. With re-rank it reaches 0.9999, but recall@100 is 0.9986.

**How the 8-bit rungs got faster than f16:**

1. **B200 (SM100):** a CUTLASS tcgen05 int8 GEMM (256×256×128 tile, 2-SM cluster) with an f16 output epilogue
   (int32 accumulate, scaled). It writes half the score bytes of cuBLAS's int32 output. The corpus is quantised with
   one scale (99.9th percentile of per-vector max |x|), so ranking needs only a per-query factor, applied in a
   vectorised filter.
2. **Small batches (≤ 8 queries):** a dp4a kernel, one warp per corpus row, with scales applied inline. cuBLAS's int8
   GEMV was slower than f16.
3. **Re-rank:** exact f16 × f16 → f32 on the device over the top 200. In speed mode it reads a device-resident f16
   copy; in low-memory mode, mapped pinned host memory.
4. **RTX PRO 6000:** the large-batch path from 128 queries (measured; it also speeds up f16 at 256: 7.16 → 5.86 ms).

**SM120 (RTX Blackwell):** CUTLASS 2.x int8 mma.sync GEMM (128×256×64 tile, 3 stages) with an alpha-only f16
epilogue. This closed the RTX gaps: plain int8 at 512 queries went from 9.8 to 7.0 ms against f16's 9.25, and
int8 + re-rank at 256 from 8.7 to 3.8 ms against 5.9.

**e-means assignment** (updates always f32; gate: objective within 0.1% of f32):

| Rung | Objective | vs f32 | B200 | RTX PRO 6000 | Gate |
|---|---:|---:|---:|---:|---|
| f32 | 0.244644 | — | 1.64-1.79 s | 1.73 s | — |
| TF32 | 0.244644 | 0.000% | 0.52-0.78 s | 0.81 s | pass |
| f16 | 0.244651 | +0.003% | 0.47 s | 0.64 s | pass |
| bf16 | 0.244668 | +0.010% | 0.49 s | 0.66 s | pass (not recommended: no faster than f16) |
| **int8 (per-row scales)** | **0.244612** | **-0.013%** | **0.42 s** | **0.57 s** | **pass, fastest** |
| FP8 e4m3 | 0.245029 | +0.157% | 0.43 s | 0.57 s | fail |

## CPU precision ladder (32 threads, same node, FAISS 1.16.0 references)

AMD EPYC 9655P (Zen 5, AVX-512 VNNI), k = 4096. Latency is 1 thread; batch is 32 threads:

| Rung | overlap@10 at nprobe 128 | Peak RSS | Latency, ms (nprobe 8 / 32 / 128) | Batch q/s (nprobe 8 / 32 / 128) |
|---|---:|---:|---|---|
| flashcluster f16 | 0.9738 | 4.40 GB | 0.48 / 1.32 / 4.90 | 51.5k / 27.6k / 9.6k |
| **flashcluster int8 (VNNI batch, AVX2 single row)** | 0.9577 | **4.16 GB** | **0.28 / 0.69 / 2.03** | **77.7k / 30.1k** / 9.6k |
| **flashcluster int8 + exact re-rank 200** (mmap f16) | **0.9738** | 4.20 GB | 0.51 / **0.81 / 2.35** | **54.2k** / 19.8k / 6.4k |
| FAISS IVF-SQ8 | 0.9677 | 4.76 GB | 0.40 / 0.98 / 2.94 | 32.7k / 12.2k / 3.2k |
| FAISS IVF-SQ8 + IndexRefineFlat (×2) | 0.9731 | 14.76 GB | 0.46 / 1.04 / 3.03 | 27.9k / 10.2k / 3.3k |
| FAISS IVF-SQfp16 | 0.9731 | 8.83 GB | 0.52 / 1.46 / 5.26 | 20.9k / 6.4k / 1.8k |
| FAISS IVF-bf16 | 0.9677 | 8.84 GB | 0.74 / 2.38 / 8.92 | 21.2k / 6.0k / 1.6k |

- **int8 vs FAISS's 8-bit index:** int8 beats IVF-SQ8 at every point on speed and memory. SQ8's per-dimension
  quantisation is more accurate plain (0.968 vs 0.958); with re-rank both are exact.
- **int8 + re-rank vs SQ8 + refine:** faster at every point except single-query nprobe 8 (0.51 vs 0.46 ms), at 3.5×
  less memory.
- **Against f16:** int8 + re-rank is slower than f16 for large batches at nprobe 32-128, because the shortlist rows
  come from the memory-mapped file.
- **Build memory:** int8 is quantised in place inside the f16 allocation, so build peak is one copy.
- **Zen 2 (EPYC 7702, AVX2), same pattern:** int8 1.08 ms / 33k q/s against FAISS SQ8 1.88 ms / 12k q/s at
  nprobe 8.

## CPU (IVF, k = 4096, same node, 32 threads; FAISS 1.16.0, IndexIVFFlat / IndexIVFScalarQuantizer fp16)

Latency is one query per call on 1 thread. Throughput is the full query batch on 32 threads. Recall is identical at each
nprobe (flashcluster overlap@10 is ≥ FAISS's by ≤ 0.0007, from tie order).

**AMD EPYC 9655P (Zen 5, AVX-512):**

| nprobe | R@10 | Latency, ms: flashcluster / FAISS Flat / FAISS SQfp16 | Batch q/s: flashcluster / Flat / SQfp16 |
|---:|---:|---|---|
| 8 | 0.953 | **0.36** / 0.59 / 0.48 | **55,812** / 11,575 / 20,354 |
| 16 | 0.962 | **0.58** / 0.99 / 0.79 | **39,834** / 6,354 / 11,170 |
| 32 | 0.969 | **1.01** / 1.79 / 1.39 | **26,268** / 3,293 / 6,418 |
| 64 | 0.971 | **1.87** / 3.35 / 2.52 | **15,906** / 1,689 / 3,332 |
| 128 | 0.973 | **3.61** / 6.44 / 6.55 | **9,551** / 861 / 1,586 |

**AMD EPYC 7702 (Zen 2, AVX2):**

| nprobe | Latency, ms: flashcluster / FAISS Flat / FAISS SQfp16 | Batch q/s: flashcluster / Flat / SQfp16 |
|---:|---|---|
| 8 | **1.16** / 1.49 / 2.45 | **19,363** / 4,285 / 7,527 |
| 64 | **5.29** / 8.30 / 12.39 | **8,076** / 618 / 1,149 |
| 128 | **9.87** / 14.8 / 23.2 | **4,866** / 310 / 558 |

**Notes:**

- **AVX-512 vs AVX2** (same Zen 5 node): 3-14% lower latency, about equal batch throughput. The scan is memory-bound.
- **Build from shared centroids:** 4.3 s (Zen 5) / 10.7 s (Zen 2), against 21-45 s for FAISS.
- **Peak RSS:**
  - An earlier run of the same engine on Zen 2 peaked at 5.0 GB, against 13.5 GB for FAISS Flat and 8.8 GB for
    SQfp16.
  - These flashcluster runs measured 7.9-8.5 GB, because the benchmark held a second copy of the corpus. That copy is
    now removed, and the Zen 3 (Milan) run re-measures it.
- **Clustering on CPU, 32 threads, k = 4096 on 2.43M:**
  - e-means (quality law, 5 epochs): 25.8 s (Zen 5) / 59.9 s (Zen 2), objective 0.24463.
  - Lloyd (25 iterations, 256 points per centroid): 43.7 s / 109 s, objective 0.24645.
