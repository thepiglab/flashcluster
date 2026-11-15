// flashcluster GPU engine (feature `cuda`): exact inner-product search and k-means/e-means batch statistics over a
// device-resident f16 matrix. C ABI for the Rust side (src/gpu.rs). One process-wide context (device, stream, cuBLAS).
//
// Exact search: Q (nq x d) . X^T in 64k-column chunks with cublasGemmEx (f16 inputs, f32 accumulation and output);
// two-stage device top-k per chunk: CUB block radix sort of each 4096-column segment (stage 1), then the running k
// plus all segment winners in one more block sort (stage 2). Equal scores keep their input order (the host wrapper re-sorts each
// row by (score desc, id asc)).
// Batch statistics (e-means / Lloyd): gather the batch rows (f16 -> f32), SGEMM against the f32 centroids (exact f32
// math unless TF32 is enabled), warp argmin of |c|^2 - 2 x.c, per-cluster sums and counts with atomics.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <mma.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
cudaStream_t g_stream = nullptr;
cublasHandle_t g_blas = nullptr;
cublasLtHandle_t g_lt = nullptr;
// storage / compute precision of the searched corpus (fc_upload_variant); e-means assignment precision (fc_em_init)
enum { V_F16 = 0, V_BF16 = 1, V_F16ACC = 2, V_INT8 = 3, V_FP8 = 4, V_INT8G = 5 };
#ifdef FC_HAVE_CUTLASS
extern "C" int fc_cutlass80_int8_available();
extern "C" int fc_cutlass80_int8_gemm(const void*, const void*, void*, int, int, int, float, cudaStream_t);
extern "C" int fc_cutlass_int8_available();
extern "C" int fc_cutlass_int8_gemm(const void*, const void*, void*, int, int, int, float, cudaStream_t);
#endif
float g_sg = 1.f;          // V_INT8G: one corpus scale (99.9th percentile of per-vector max |x|, / 127)
constexpr float G_ALPHA = 1.f / 256;   // CUTLASS epilogue: f16 = acc / 256 (keeps int8 dot products in f16 range)
static int cutlass_kind();
static bool cutlass_ok() { return cutlass_kind() != 0; }
static int cutlass_kind() {
#ifdef FC_HAVE_CUTLASS
  // 1: SM100 tcgen05 kernel, 2: SM120 mma.sync kernel, 0: cuBLAS
  static int v = [] {
    const char* e = getenv("FLASHCLUSTER_CUTLASS_GEMM");
    if (e && !atoi(e)) return 0;
    return fc_cutlass_int8_available() ? 1 : fc_cutlass80_int8_available() ? 2 : 0;
  }();
  return v;
#else
  return false;
#endif
}
int g_var = V_F16;
int g_em_mode = 0;   // 0 f32, 1 tf32, 2 f16, 3 bf16, 4 int8, 5 fp8 assignment inputs (updates always f32)
char g_err[512] = {0};

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      snprintf(g_err, sizeof g_err, "%s:%d %s", __FILE__, __LINE__, cudaGetErrorString(e_));      \
      return -1;                                                                                  \
    }                                                                                             \
  } while (0)
#define CB(x)                                                                                     \
  do {                                                                                            \
    cublasStatus_t s_ = (x);                                                                      \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                            \
      snprintf(g_err, sizeof g_err, "%s:%d cublas status %d", __FILE__, __LINE__, (int)s_);       \
      return -1;                                                                                  \
    }                                                                                             \
  } while (0)

// ---------------------------------------------------------------- exact search top-k
constexpr int SORT_THREADS = 256;
constexpr int SORT_ITEMS = 16;                      // 4096 keys per block sort
constexpr int SORT_N = SORT_THREADS * SORT_ITEMS;
constexpr int SEG = SORT_N;                         // stage 1: columns per segment
constexpr int CHUNK_SEGS = 16;                      // first GEMM chunk = 16 segments = 65536 columns
constexpr int CAND = 3584;                          // filtered candidates per row per chunk (k + CAND <= SORT_N)

using Sort = cub::BlockRadixSort<float, SORT_THREADS, SORT_ITEMS, int32_t>;

// Stage 1, block (row r, segment g): top-k of scores[r, g*SEG .. ) -> cand[r][g][0..k)
__global__ void __launch_bounds__(SORT_THREADS) seg_topk(const float* __restrict__ scores, int64_t ld, int cols,
                                                         int64_t base, int k, float* __restrict__ cs,
                                                         int32_t* __restrict__ ci) {
  __shared__ typename Sort::TempStorage tmp;
  const int r = blockIdx.x, g = blockIdx.y, nseg = gridDim.y;
  const int c0 = g * SEG;
  float key[SORT_ITEMS];
  int32_t val[SORT_ITEMS];
  const float* row = scores + (int64_t)r * ld;
#pragma unroll
  for (int i = 0; i < SORT_ITEMS; ++i) {
    int c = c0 + threadIdx.x * SORT_ITEMS + i;
    bool ok = c < cols;
    key[i] = ok ? row[c] : -INFINITY;
    val[i] = ok ? (int32_t)(base + c) : INT32_MAX;
  }
  Sort(tmp).SortDescendingBlockedToStriped(key, val);
  float* os = cs + ((int64_t)r * nseg + g) * k;
  int32_t* oi = ci + ((int64_t)r * nseg + g) * k;
#pragma unroll
  for (int i = 0; i < SORT_ITEMS; ++i) {
    int rank = i * SORT_THREADS + threadIdx.x;
    if (rank < k) { os[rank] = key[i]; oi[rank] = val[i]; }
  }
}

// Merge, block r: running top-k + candidates cand[r][0..cnt) (stride `stride`) -> new running top-k; updates the
// row threshold tau (k-th best, or -inf while fewer than k are known) and resets the count. Two sizes: 4096 slots, and
// 1024 slots when every row's candidates fit (the common case after the first chunk).
template <int T, int I>
__global__ void __launch_bounds__(T) merge_topk_t(const float* __restrict__ cs, const int32_t* __restrict__ ci,
                                                  const int* __restrict__ cnt_in, int fixed_cnt, int stride, int k,
                                                  float* __restrict__ run_s, int32_t* __restrict__ run_i,
                                                  float* __restrict__ tau, int* __restrict__ cnt_reset) {
  using S = cub::BlockRadixSort<float, T, I, int32_t>;
  __shared__ typename S::TempStorage tmp;
  const int r = blockIdx.x;
  const int ncand = cnt_in ? min(cnt_in[r], stride) : fixed_cnt;
  float key[I];
  int32_t val[I];
#pragma unroll
  for (int i = 0; i < I; ++i) {
    int slot = threadIdx.x * I + i;
    if (slot < k) {
      key[i] = run_s[(int64_t)r * k + slot];
      val[i] = run_i[(int64_t)r * k + slot];
    } else if (slot - k < ncand) {
      key[i] = cs[(int64_t)r * stride + slot - k];
      val[i] = ci[(int64_t)r * stride + slot - k];
    } else {
      key[i] = -INFINITY;
      val[i] = INT32_MAX;
    }
  }
  S(tmp).SortDescendingBlockedToStriped(key, val);
#pragma unroll
  for (int i = 0; i < I; ++i) {
    int rank = i * T + threadIdx.x;
    if (rank < k) {
      run_s[(int64_t)r * k + rank] = key[i];
      run_i[(int64_t)r * k + rank] = val[i];
    }
    if (rank == k - 1) tau[r] = key[i];
  }
  if (cnt_reset && threadIdx.x == 0) cnt_reset[r] = 0;
}
#define merge_topk merge_topk_t<SORT_THREADS, SORT_ITEMS>
#define merge_topk_small merge_topk_t<128, 8>
constexpr int SMALL_N = 128 * 8;

// Filter, grid (row, column blocks): keep scores above the row threshold (float4 loads; cols % 4 == 0 except the
// tail); overflow raises stat[0], stat[1] = max candidates in any row.
__device__ __forceinline__ void keep(int r, float v, int64_t id, float t, int stride, float* cs, int32_t* ci, int* cnt,
                                     int* stat) {
  if (v > t) {
    int slot = atomicAdd(&cnt[r], 1);
    if (slot < stride) {
      cs[(int64_t)r * stride + slot] = v;
      ci[(int64_t)r * stride + slot] = (int32_t)id;
    } else {
      stat[0] = 1;
    }
    atomicMax(&stat[1], slot + 1);
  }
}

__global__ void filter_above(const float* __restrict__ scores, int64_t ld, int cols, int64_t base,
                             const float* __restrict__ tau, int stride, float* __restrict__ cs,
                             int32_t* __restrict__ ci, int* __restrict__ cnt, int* __restrict__ stat) {
  const int r = blockIdx.x;
  const float t = tau[r];
  const float* row = scores + (int64_t)r * ld;
  if (ld & 3) {  // rows not 16-byte aligned: scalar path
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < cols; c += gridDim.y * blockDim.x)
      keep(r, row[c], base + c, t, stride, cs, ci, cnt, stat);
    return;
  }
  const int c4 = cols / 4;
  const float4* row4 = reinterpret_cast<const float4*>(row);
  for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < c4; j += gridDim.y * blockDim.x) {
    float4 v = __ldg(row4 + j);
    if (fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)) <= t) continue;
    int64_t id = base + 4 * (int64_t)j;
    keep(r, v.x, id, t, stride, cs, ci, cnt, stat);
    keep(r, v.y, id + 1, t, stride, cs, ci, cnt, stat);
    keep(r, v.z, id + 2, t, stride, cs, ci, cnt, stat);
    keep(r, v.w, id + 3, t, stride, cs, ci, cnt, stat);
  }
  if (blockIdx.y == 0)
    for (int c = 4 * c4 + threadIdx.x; c < cols; c += blockDim.x) keep(r, row[c], base + c, t, stride, cs, ci, cnt, stat);
}

// f16-score filter: scores are the GEMM output rounded to f16; keep ids whose f16 score > tau - MARGIN (the margin
// covers f16 rounding, so no true top-k candidate is lost); exact f32 scores are recomputed by `rescore`.
constexpr float MARGIN = 2e-3f;
__global__ void filter_above16(const __half* __restrict__ scores, int64_t ld, int cols, int64_t base,
                               const float* __restrict__ tau, int stride, int32_t* __restrict__ ci,
                               int* __restrict__ cnt, int* __restrict__ stat) {
  const int r = blockIdx.x;
  const float t = tau[r] - MARGIN;
  const __half* row = scores + (int64_t)r * ld;
  auto keep16 = [&](float v, int64_t id) {
    if (v > t) {
      int slot = atomicAdd(&cnt[r], 1);
      if (slot < stride) ci[(int64_t)r * stride + slot] = (int32_t)id;
      else stat[0] = 1;
      atomicMax(&stat[1], slot + 1);
    }
  };
  if (ld & 7) {
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < cols; c += gridDim.y * blockDim.x) keep16(__half2float(row[c]), base + c);
    return;
  }
  const int c8 = cols / 8;
  const uint4* row8 = reinterpret_cast<const uint4*>(row);
  for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < c8; j += gridDim.y * blockDim.x) {
    uint4 u = __ldg(row8 + j);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
    float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), e = __half22float2(h[3]);
    float m = fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(b.x, b.y)), fmaxf(fmaxf(c.x, c.y), fmaxf(e.x, e.y)));
    if (m <= t) continue;
    int64_t id = base + 8 * (int64_t)j;
    keep16(a.x, id); keep16(a.y, id + 1); keep16(b.x, id + 2); keep16(b.y, id + 3);
    keep16(c.x, id + 4); keep16(c.y, id + 5); keep16(e.x, id + 6); keep16(e.y, id + 7);
  }
  if (blockIdx.y == 0)
    for (int c = 8 * c8 + threadIdx.x; c < cols; c += blockDim.x) keep16(__half2float(row[c]), base + c);
}

// Exact f32 scores (f16 inputs, f32 accumulation, as the GEMM) for each row's candidates; one warp per candidate.
__global__ void rescore(const __half* __restrict__ X, const __half* __restrict__ Q, int d, const int32_t* __restrict__ ci,
                        const int* __restrict__ cnt, int stride, float* __restrict__ cs) {
  const int r = blockIdx.x;
  const int n = min(cnt[r], stride);
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, nw = blockDim.x / 32;
  const __half2* q2 = reinterpret_cast<const __half2*>(Q + (int64_t)r * d);
  for (int i = warp; i < n; i += nw) {
    const __half2* x2 = reinterpret_cast<const __half2*>(X + (int64_t)ci[(int64_t)r * stride + i] * d);
    float s = 0.f;
    for (int t = lane; t < d / 2; t += 32) {
      float2 a = __half22float2(q2[t]), b = __half22float2(x2[t]);
      s = fmaf(a.x, b.x, s);
      s = fmaf(a.y, b.y, s);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (!lane) cs[(int64_t)r * stride + i] = s;
  }
}

__global__ void f32_to_f16(const float* __restrict__ a, __half* __restrict__ b, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2half_rn(a[i]);
}

// ---------------------------------------------------------------- batch statistics
__global__ void gather_f16_to_f32(const __half* __restrict__ x, const int32_t* __restrict__ rows, int64_t m, int d,
                                  float* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= m * d) return;
  int64_t r = i / d, c = i % d;
  int64_t src = rows ? (int64_t)rows[r] : r;
  out[i] = __half2float(x[src * d + c]);
}

__global__ void row_sqnorm(const float* __restrict__ a, int64_t n, int d, float* __restrict__ out) {
  int64_t r = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= n) return;
  float s = 0.f;
  for (int c = lane; c < d; c += 32) { float v = a[r * d + c]; s += v * v; }
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
  if (!lane) out[r] = s;
}

// one warp per row: argmin_j c2[j] - 2 s[r,j]; dist = max(0, x2 + min)
__global__ void argmin_l2(const float* __restrict__ s, int64_t m, int k, const float* __restrict__ c2,
                          const float* __restrict__ x2, int32_t* __restrict__ lab, float* __restrict__ dist) {
  int64_t r = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= m) return;
  float best = INFINITY;
  int bj = 0;
  for (int j = lane; j < k; j += 32) {
    float v = c2[j] - 2.f * s[r * k + j];
    if (v < best || (v == best && j < bj)) { best = v; bj = j; }
  }
  for (int o = 16; o; o >>= 1) {
    float ob = __shfl_down_sync(0xffffffff, best, o);
    int oj = __shfl_down_sync(0xffffffff, bj, o);
    if (ob < best || (ob == best && oj < bj)) { best = ob; bj = oj; }
  }
  if (!lane) { lab[r] = bj; dist[r] = fmaxf(0.f, x2[r] + best); }
}

__global__ void accumulate(const float* __restrict__ xb, int64_t m, int d, const int32_t* __restrict__ lab,
                           float* __restrict__ sums, uint32_t* __restrict__ counts) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= m * d) return;
  int64_t r = i / d, c = i % d;
  int j = lab[r];
  atomicAdd(&sums[(int64_t)j * d + c], xb[i]);
  if (c == 0) atomicAdd(&counts[j], 1u);
}


// ---------------------------------------------------------------- device-resident e-means engine
__global__ void add_inplace(float* __restrict__ a, const float* __restrict__ b, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

// c_j += w_j (sums_j / n_j - c_j) for n_j > 0 (w_j == 1: c_j = mean_j)
__global__ void em_update(float* __restrict__ c, const float* __restrict__ sums, const uint32_t* __restrict__ cnt,
                          const float* __restrict__ w, int k, int d) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)k * d) return;
  int j = (int)(i / d);
  uint32_t nj = cnt[j];
  if (!nj) return;
  float mean = sums[i] * (1.0f / (float)nj);
  float wj = w[j];
  c[i] = wj == 1.0f ? mean : c[i] + wj * (mean - c[i]);
}

// one block per (src, dst) pair: c_src *= 1 - off, c_dst = old c_src * (1 + off); collapsed rows nudged in coord 0
__global__ void em_split(float* __restrict__ c, const int32_t* __restrict__ pairs, int d) {
  const int src = pairs[2 * blockIdx.x], dst = pairs[2 * blockIdx.x + 1];
  int same = 1;
  for (int t = threadIdx.x; t < d; t += blockDim.x) {
    float off = (t % 2 == 0) ? 1e-4f : -1e-4f;
    float v = c[(int64_t)src * d + t];
    float a = v * (1.0f - off), b = v * (1.0f + off);
    same &= (a == b);
    c[(int64_t)src * d + t] = a;
    c[(int64_t)dst * d + t] = b;
  }
  same = __syncthreads_and(same);
  if (same && threadIdx.x == 0) {
    float v = c[(int64_t)dst * d];
    float up = nextafterf(v, INFINITY);
    c[(int64_t)dst * d] = isfinite(up) ? up : -nextafterf(-v, INFINITY);
  }
}

__global__ void em_close(float* __restrict__ c, const float* __restrict__ close, const double* __restrict__ pc, int k, int d) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)k * d) return;
  double n = pc[i / d];
  if (n > 0) c[i] = close[i] * (float)(1.0 / n);
}


// ---------------------------------------------------------------- precision variants
// one warp per row: per-row max-abs scale, then quantise (int8: /127, fp8 e4m3: /448) or convert (bf16)
__global__ void quantize_rows(const __half* __restrict__ x, int64_t n, int d, int var, void* __restrict__ out,
                              float* __restrict__ scale) {
  int64_t r = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= n) return;
  const __half* row = x + r * d;
  if (var == V_BF16) {
    __nv_bfloat16* o = (__nv_bfloat16*)out + r * d;
    for (int t = lane; t < d; t += 32) o[t] = __float2bfloat16_rn(__half2float(row[t]));
    return;
  }
  float m = 0.f;
  for (int t = lane; t < d; t += 32) m = fmaxf(m, fabsf(__half2float(row[t])));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  float s = m > 0 ? m / (var == V_INT8 ? 127.f : 448.f) : 1.f;
  if (!lane) scale[r] = s;
  float inv = 1.f / s;
  for (int t = lane; t < d; t += 32) {
    float v = __half2float(row[t]) * inv;
    if (var == V_INT8) ((int8_t*)out)[r * d + t] = (int8_t)max(-127, min(127, __float2int_rn(v)));
    else ((__nv_fp8_e4m3*)out)[r * d + t] = __nv_fp8_e4m3(v);
  }
}

// in place: int32 (int8 GEMM) or f32 (fp8 GEMM) raw scores -> f32 scores * qscale[row] * xscale[base + col]
__global__ void apply_scales(float* __restrict__ s, int64_t nq, int cols, int64_t base, int is_int,
                             const float* __restrict__ qs, const float* __restrict__ xs) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= nq * cols) return;
  int64_t r = i / cols, c = i % cols;
  float v = is_int ? (float)reinterpret_cast<int32_t*>(s)[i] : s[i];
  s[i] = v * qs[r] * xs[base + c];
}

__global__ void half_to_f32(const __half* __restrict__ a, float* __restrict__ b, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = __half2float(a[i]);
}

__global__ void f32_to_bf16(const float* __restrict__ a, __nv_bfloat16* __restrict__ b, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2bfloat16_rn(a[i]);
}

__global__ void gather_f16(const __half* __restrict__ x, const int32_t* __restrict__ rows, int64_t m, int d,
                           __half* __restrict__ out, int to_bf16) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= m * d) return;
  __half v = x[(int64_t)rows[i / d] * d + i % d];
  if (to_bf16) ((__nv_bfloat16*)out)[i] = __float2bfloat16_rn(__half2float(v));
  else out[i] = v;
}


// Variant score filter: raw GEMM output of type T (half, bf16 or int32) -> score (times qs[row] * xs[col] when scales
// are given); keep id and score when score > tau (approximate variants: no margin, no rescore).
template <typename T> __device__ __forceinline__ float sc_to_f(T v);
template <> __device__ __forceinline__ float sc_to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float sc_to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float sc_to_f<int32_t>(int32_t v) { return (float)v; }
template <> __device__ __forceinline__ float sc_to_f<float>(float v) { return v; }

template <typename T>
__global__ void filter_var(const T* __restrict__ scores, int64_t ld, int cols, int64_t base, const float* __restrict__ tau,
                           int stride, float* __restrict__ cs, int32_t* __restrict__ ci, int* __restrict__ cnt,
                           int* __restrict__ stat, const float* __restrict__ qs, const float* __restrict__ xs) {
  const int r = blockIdx.x;
  const float t = tau[r];
  const float qr = qs ? qs[r] : 1.f;
  const T* row = scores + (int64_t)r * ld;
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < cols; c += gridDim.y * blockDim.x) {
    float v = sc_to_f<T>(row[c]);
    if (qs) v *= qr * (xs ? xs[base + c] : 1.f);
    if (v > t) {
      int slot = atomicAdd(&cnt[r], 1);
      if (slot < stride) { cs[(int64_t)r * stride + slot] = v; ci[(int64_t)r * stride + slot] = (int32_t)(base + c); }
      else stat[0] = 1;
      atomicMax(&stat[1], slot + 1);
    }
  }
}

// Exact re-rank from pinned host f16 rows (mapped into the device address space): cs[r][i] = q_r . x_{ids[r][i]}.
__global__ void rerank_host(const __half* __restrict__ Xh, const __half* __restrict__ Q, int d, const int32_t* __restrict__ ids,
                            int L, float* __restrict__ cs) {
  const int r = blockIdx.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, nw = blockDim.x / 32;
  const __half2* q2 = reinterpret_cast<const __half2*>(Q + (int64_t)r * d);
  for (int i = warp; i < L; i += nw) {
    int32_t id = ids[(int64_t)r * L + i];
    float s = -INFINITY;
    if (id != INT32_MAX) {
      const uint4* x4 = reinterpret_cast<const uint4*>(Xh + (int64_t)id * d);   // 8 halves per 16-byte load
      s = 0.f;
      for (int t = lane; t < d / 8; t += 32) {
        uint4 u = x4[t];
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 a = __half22float2(q2[4 * t + e]), b = __half22float2(h[e]);
          s = fmaf(a.x, b.x, s);
          s = fmaf(a.y, b.y, s);
        }
      }
      for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    }
    if (!lane) cs[(int64_t)r * L + i] = s;
  }
}


// argmin with 8-bit assignment scores: dot = raw[r,j] * cs[j] * xs[r] (raw int32 or f32)
__global__ void argmin_l2_q(const void* __restrict__ s, int is_int, int64_t m, int k, const float* __restrict__ c2,
                            const float* __restrict__ x2, const float* __restrict__ cs, const float* __restrict__ xs,
                            int32_t* __restrict__ lab, float* __restrict__ dist) {
  int64_t r = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= m) return;
  float best = INFINITY, xr = xs[r];
  int bj = 0;
  for (int j = lane; j < k; j += 32) {
    float raw = is_int ? (float)((const int32_t*)s)[r * k + j] : ((const float*)s)[r * k + j];
    float v = c2[j] - 2.f * raw * cs[j] * xr;
    if (v < best || (v == best && j < bj)) { best = v; bj = j; }
  }
  for (int o = 16; o; o >>= 1) {
    float ob = __shfl_down_sync(0xffffffff, best, o);
    int oj = __shfl_down_sync(0xffffffff, bj, o);
    if (ob < best || (ob == best && oj < bj)) { best = ob; bj = oj; }
  }
  if (!lane) { lab[r] = bj; dist[r] = fmaxf(0.f, x2[r] + best); }
}


// ---------------------------------------------------------------- fused tensor-core scoring + threshold filter
// Block tile: FBM corpus rows x FBN queries, k-step FBK, 8 warps (4 along rows x 2 along queries, 32 x 64 each).
// Scores never leave the SM: each 16x16 accumulator fragment is staged in a per-warp 1 KiB buffer, scaled, compared
// with the query's threshold, and only passing (score, id) pairs are appended to that query's candidate list.
// Grid: x = query tiles (fastest), y = corpus tiles, so a corpus tile is reused from L2 by every query tile.
constexpr int FBM = 128, FBN = 128, FBK = 64;
template <typename TIn, typename TAcc>
__global__ void __launch_bounds__(256) fused_score_filter(const TIn* __restrict__ X, const TIn* __restrict__ Q, int64_t c0,
    int cols, int nq, int d, const float* __restrict__ xs, const float* __restrict__ qs, const float* __restrict__ tau,
    int stride, float* __restrict__ cs, int32_t* __restrict__ ci, int* __restrict__ cnt, int* __restrict__ stat) {
  using namespace nvcuda;
  __shared__ __align__(16) TIn As[FBM * FBK];
  __shared__ __align__(16) TIn Bs[FBN * FBK];
  __shared__ __align__(16) TAcc Cs[8][16 * 16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int wm = warp / 2, wn = warp % 2;                  // warp tile: rows wm*32.., queries wn*64..
  const int64_t row0 = (int64_t)blockIdx.y * FBM;          // within the chunk
  const int q0 = blockIdx.x * FBN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, TAcc> acc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], (TAcc)0);
  constexpr int VEC = 16 / sizeof(TIn);                    // elements per 16-byte load
  for (int k0 = 0; k0 < d; k0 += FBK) {
    for (int v = threadIdx.x; v < FBM * FBK / VEC; v += 256) {
      int r = v / (FBK / VEC), c = (v % (FBK / VEC)) * VEC;
      int64_t gr = row0 + r;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (gr < cols && k0 + c < d) val = *reinterpret_cast<const uint4*>(X + (c0 + gr) * d + k0 + c);
      *reinterpret_cast<uint4*>(As + r * FBK + c) = val;
    }
    for (int v = threadIdx.x; v < FBN * FBK / VEC; v += 256) {
      int r = v / (FBK / VEC), c = (v % (FBK / VEC)) * VEC;
      int gq = q0 + r;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (gq < nq && k0 + c < d) val = *reinterpret_cast<const uint4*>(Q + (int64_t)gq * d + k0 + c);
      *reinterpret_cast<uint4*>(Bs + r * FBK + c) = val;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < FBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, TIn, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, TIn, wmma::col_major> b[4];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * FBK + kk, FBK);
#pragma unroll
      for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(b[j], Bs + (wn * 64 + j * 16) * FBK + kk, FBK);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  TAcc* cw = Cs[warp];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wmma::store_matrix_sync(cw, acc[i][j], 16, wmma::mem_row_major);   // [row][query]
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int rr = e / 16, qq = e % 16;
        int64_t gr = row0 + wm * 32 + i * 16 + rr;
        int gq = q0 + wn * 64 + j * 16 + qq;
        if (gr >= cols || gq >= nq) continue;
        float v = (float)cw[e];
        if (xs) v *= xs[c0 + gr] * qs[gq];
        if (v > tau[gq]) {
          int slot = atomicAdd(&cnt[gq], 1);
          if (slot < stride) { cs[(int64_t)gq * stride + slot] = v; ci[(int64_t)gq * stride + slot] = (int32_t)(c0 + gr); }
          else stat[0] = 1;
          atomicMax(&stat[1], slot + 1);
        }
      }
      __syncwarp();
    }
}


// Merge without host sync: each block picks a 1024-slot or 4096-slot sort from its own candidate count.
template <int I>
__device__ __forceinline__ void merge_rows(void* tmp, int r, const float* cs, const int32_t* ci, int ncand, int stride, int k,
                                           float* run_s, int32_t* run_i, float* tau) {
  using S = cub::BlockRadixSort<float, 256, I, int32_t>;
  float key[I];
  int32_t val[I];
#pragma unroll
  for (int i = 0; i < I; ++i) {
    int slot = threadIdx.x * I + i;
    if (slot < k) { key[i] = run_s[(int64_t)r * k + slot]; val[i] = run_i[(int64_t)r * k + slot]; }
    else if (slot - k < ncand) { key[i] = cs[(int64_t)r * stride + slot - k]; val[i] = ci[(int64_t)r * stride + slot - k]; }
    else { key[i] = -INFINITY; val[i] = INT32_MAX; }
  }
  S(*reinterpret_cast<typename S::TempStorage*>(tmp)).SortDescendingBlockedToStriped(key, val);
#pragma unroll
  for (int i = 0; i < I; ++i) {
    int rank = i * 256 + threadIdx.x;
    if (rank < k) { run_s[(int64_t)r * k + rank] = key[i]; run_i[(int64_t)r * k + rank] = val[i]; }
    if (rank == k - 1) tau[r] = key[i];
  }
}

__global__ void __launch_bounds__(256) merge_adaptive(const float* __restrict__ cs, const int32_t* __restrict__ ci,
                                                      int* __restrict__ cnt, int stride, int k, float* __restrict__ run_s,
                                                      int32_t* __restrict__ run_i, float* __restrict__ tau) {
  __shared__ union { typename cub::BlockRadixSort<float, 256, 4, int32_t>::TempStorage a;
                     typename cub::BlockRadixSort<float, 256, 16, int32_t>::TempStorage b; } tmp;
  const int r = blockIdx.x;
  const int ncand = min(cnt[r], stride);
  if (ncand > 0) {
    if (ncand + k <= 1024) merge_rows<4>(&tmp, r, cs, ci, ncand, stride, k, run_s, run_i, tau);
    else merge_rows<16>(&tmp, r, cs, ci, ncand, stride, k, run_s, run_i, tau);
  }
  __syncthreads();
  if (threadIdx.x == 0) cnt[r] = 0;
}


__global__ void row_maxabs(const __half* __restrict__ x, int64_t n, int d, float* __restrict__ out) {
  int64_t r = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= n) return;
  float m = 0.f;
  for (int t = lane; t < d; t += 32) m = fmaxf(m, fabsf(__half2float(x[r * d + t])));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  if (!lane) out[r] = m;
}
__global__ void quantize_fixed(const __half* __restrict__ x, int64_t n, int d, float inv, int8_t* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n * d) out[i] = (int8_t)max(-127, min(127, __float2int_rn(__half2float(x[i]) * inv)));
}
__global__ void fill_f32(float* __restrict__ a, int64_t n, float v) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) a[i] = v;
}
__global__ void scale_f32(const float* __restrict__ a, float* __restrict__ b, int64_t n, float v) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i] * v;
}


// Vectorised f16-score filter with a per-query scale (int8g on CUTLASS): keep (score, id) when f16 * qs[row] > tau.
__global__ void filter16_scaled(const __half* __restrict__ scores, int64_t ld, int cols, int64_t base,
                                const float* __restrict__ tau, const float* __restrict__ qs, int stride,
                                float* __restrict__ cs, int32_t* __restrict__ ci, int* __restrict__ cnt, int* __restrict__ stat) {
  const int r = blockIdx.x;
  const float q = qs[r];
  const float t = tau[r] / q;     // compare raw f16 against tau in raw units
  const __half* row = scores + (int64_t)r * ld;
  auto keep = [&](float v, int64_t id) {
    if (v > t) {
      int slot = atomicAdd(&cnt[r], 1);
      if (slot < stride) { cs[(int64_t)r * stride + slot] = v * q; ci[(int64_t)r * stride + slot] = (int32_t)id; }
      else stat[0] = 1;
      atomicMax(&stat[1], slot + 1);
    }
  };
  if (ld & 7) {
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < cols; c += gridDim.y * blockDim.x) keep(__half2float(row[c]), base + c);
    return;
  }
  const int c8 = cols / 8;
  const uint4* row8 = reinterpret_cast<const uint4*>(row);
  for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < c8; j += gridDim.y * blockDim.x) {
    uint4 u = __ldg(row8 + j);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
    float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), e = __half22float2(h[3]);
    float m = fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(b.x, b.y)), fmaxf(fmaxf(c.x, c.y), fmaxf(e.x, e.y)));
    if (m <= t) continue;
    int64_t id = base + 8 * (int64_t)j;
    keep(a.x, id); keep(a.y, id + 1); keep(b.x, id + 2); keep(b.y, id + 3);
    keep(c.x, id + 4); keep(c.y, id + 5); keep(e.x, id + 6); keep(e.y, id + 7);
  }
  if (blockIdx.y == 0)
    for (int c = 8 * c8 + threadIdx.x; c < cols; c += blockDim.x) keep(__half2float(row[c]), base + c);
}


// Small-batch int8 scores (nq <= 8): one warp per corpus row, 16-byte loads, dp4a, scales applied inline.
// out[q][c] = (x_c . q) * xs[c0 + c] * qs[q]   (row-major nq x cols)
__global__ void int8_gemv(const int8_t* __restrict__ X, int64_t c0, int cols, int d, const int8_t* __restrict__ Q, int nq,
                          const float* __restrict__ xs, const float* __restrict__ qs, float* __restrict__ out) {
  extern __shared__ int4 qsh[];                      // nq x d bytes
  const int nv = d / 16;
  for (int i = threadIdx.x; i < nq * nv; i += blockDim.x) qsh[i] = reinterpret_cast<const int4*>(Q)[i];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int64_t r = (int64_t)blockIdx.x * (blockDim.x / 32) + warp;
  if (r >= cols) return;
  const int4* x = reinterpret_cast<const int4*>(X + (c0 + r) * d);
  int acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = lane; j < nv; j += 32) {
    int4 a = __ldg(x + j);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      if (q >= nq) break;
      int4 b = qsh[q * nv + j];
      acc[q] = __dp4a(a.x, b.x, acc[q]); acc[q] = __dp4a(a.y, b.y, acc[q]);
      acc[q] = __dp4a(a.z, b.z, acc[q]); acc[q] = __dp4a(a.w, b.w, acc[q]);
    }
  }
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    if (q >= nq) break;
    int v = acc[q];
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (!lane) out[(int64_t)q * cols + r] = (float)v * xs[c0 + r] * qs[q];
  }
}

struct Buf {
  void* p = nullptr;
  size_t n = 0;
  int ensure(size_t bytes) {
    if (bytes <= n) return 0;
    if (p) cudaFree(p);
    if (cudaMalloc(&p, bytes) != cudaSuccess) { p = nullptr; n = 0; return -1; }
    n = bytes;
    return 0;
  }
  ~Buf() { if (p) cudaFree(p); }
};
Buf g_rows, g_xb, g_s, g_c, g_c2, g_x2, g_lab, g_dist, g_sums, g_cnt, g_q, g_scores, g_rs, g_ri, g_cs, g_ci, g_tau, g_cnt2, g_flag, g_qf, g_C, g_close, g_w, g_pairs, g_pc, g_s16, g_xscale, g_qscale, g_qv, g_ws, g_tmp16, g_c16, g_xb16, g_c8, g_x8, g_cs8, g_xs8, g_qs2;
int g_k = 0, g_d = 0;
int g_ld16 = 0;   // leading dimension of the last gemm_s16 output (CUTLASS pads to a multiple of 8)
}  // namespace

extern "C" {

const char* fc_last_error() { return g_err; }

int fc_init(int device, int tf32) {
  CK(cudaSetDevice(device));
  if (!g_stream) CK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  if (!g_blas) {
    CB(cublasCreate(&g_blas));
    CB(cublasSetStream(g_blas, g_stream));
  }
  CB(cublasSetMathMode(g_blas, CUBLAS_DEFAULT_MATH));   // exact f32 SGEMM unless a call asks for TF32
  if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) { snprintf(g_err, sizeof g_err, "cublasLtCreate failed"); return -1; }
  (void)tf32;
  return 0;
}

int fc_device_name(char* out, int len) {
  cudaDeviceProp p;
  int dev;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&p, dev));
  snprintf(out, len, "%s (sm_%d%d, %.1f GB)", p.name, p.major, p.minor, p.totalGlobalMem / 1e9);
  return 0;
}

// Upload an f16 matrix; returns a device pointer (opaque) or null.
void* fc_upload_f16(const uint16_t* host, int64_t n, int d) {
  void* p = nullptr;
  if (cudaMalloc(&p, (size_t)n * d * 2) != cudaSuccess) return nullptr;
  if (cudaMemcpy(p, host, (size_t)n * d * 2, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(p); return nullptr; }
  return p;
}

// Upload the corpus (host f16, n x d) in a precision variant (V_F16 / V_F16ACC keep f16; V_BF16 converts; V_INT8 /
// V_FP8 quantise per vector with a stored f32 scale). Sets the variant used by fc_exact_search. Returns the device
// corpus pointer, or null.
void* fc_upload_variant(const uint16_t* host, int64_t n, int d, int var) {
  g_var = var;
  if (var == V_F16 || var == V_F16ACC) return fc_upload_f16(host, n, d);
  if (var == V_INT8G) {   // two passes: per-vector max |x| -> one corpus scale; then quantise with it
    void* out = nullptr;
    if (cudaMalloc(&out, (size_t)(n + 8) * d) != cudaSuccess || g_xscale.ensure((size_t)n * 4)) return nullptr;
    cudaMemset((int8_t*)out + n * d, 0, (size_t)8 * d);   // zero pad rows: CUTLASS runs on widths rounded up to 8
    const int64_t B = 131072;
    void* tmp = nullptr;
    if (cudaMalloc(&tmp, (size_t)B * d * 2) != cudaSuccess) { cudaFree(out); return nullptr; }
    for (int64_t s0 = 0; s0 < n; s0 += B) {
      int64_t m = std::min(B, n - s0);
      cudaMemcpy(tmp, host + s0 * d, (size_t)m * d * 2, cudaMemcpyHostToDevice);
      row_maxabs<<<(unsigned)((m + 7) / 8), 256>>>((const __half*)tmp, m, d, (float*)g_xscale.p + s0);
    }
    std::vector<float> mx(n);
    cudaMemcpy(mx.data(), g_xscale.p, (size_t)n * 4, cudaMemcpyDeviceToHost);
    std::nth_element(mx.begin(), mx.begin() + (int64_t)(0.999 * (n - 1)), mx.end());
    g_sg = std::max(mx[(int64_t)(0.999 * (n - 1))], 1e-12f) / 127.f;
    for (int64_t s0 = 0; s0 < n; s0 += B) {
      int64_t m = std::min(B, n - s0);
      cudaMemcpy(tmp, host + s0 * d, (size_t)m * d * 2, cudaMemcpyHostToDevice);
      quantize_fixed<<<(unsigned)((m * d + 255) / 256), 256>>>((const __half*)tmp, m, d, 1.f / g_sg, (int8_t*)out + s0 * d);
    }
    fill_f32<<<(unsigned)((n + 255) / 256), 256>>>((float*)g_xscale.p, n, g_sg);
    cudaDeviceSynchronize();
    cudaFree(tmp);
    if (cudaGetLastError() != cudaSuccess) { cudaFree(out); return nullptr; }
    return out;
  }
  size_t esz = var == V_BF16 ? 2 : 1;
  void* out = nullptr;
  if (cudaMalloc(&out, (size_t)n * d * esz) != cudaSuccess) return nullptr;
  if (var != V_BF16 && g_xscale.ensure((size_t)n * 4)) { cudaFree(out); return nullptr; }
  const int64_t B = 131072;
  void* tmp = nullptr;
  if (cudaMalloc(&tmp, (size_t)B * d * 2) != cudaSuccess) { cudaFree(out); return nullptr; }
  for (int64_t s0 = 0; s0 < n; s0 += B) {
    int64_t m = std::min(B, n - s0);
    cudaMemcpy(tmp, host + s0 * d, (size_t)m * d * 2, cudaMemcpyHostToDevice);
    quantize_rows<<<(unsigned)((m + 7) / 8), 256>>>((const __half*)tmp, m, d, var, (char*)out + s0 * d * esz,
                                                    var == V_BF16 ? nullptr : (float*)g_xscale.p + s0);
  }
  cudaDeviceSynchronize();
  cudaFree(tmp);
  if (cudaGetLastError() != cudaSuccess) { cudaFree(out); return nullptr; }
  return out;
}

void fc_free(void* p) { if (p) cudaFree(p); }

int fc_mem_used(int64_t* used) {
  size_t f, t;
  CK(cudaMemGetInfo(&f, &t));
  *used = (int64_t)(t - f);
  return 0;
}

// Exact top-k inner product of nq f32 queries against the device f16 matrix X (n x d).
// * First FIRST columns: f32-output GEMM and exact segment sorts give each row its top-k and threshold tau.
// * Then chunks (growing up to 8x the columns seen, bounded by a 256 MiB f16 score buffer): f16-output GEMM (half the
//   bytes of f32), keep ids whose f16 score > tau - MARGIN, rescore those exactly in f32, merge (tau rises).
// * If a row overflows CAND candidates in a chunk, that chunk is redone exactly in FIRST-column f32 pieces.

// f32 scores (row-major nq x cols, ld = cols) of the queries against corpus columns [c0, c0+cols) in the current
// precision variant; queries were prepared by prep_queries.
static int lt_fp8(const void* A, const void* B, void* D, int m, int n, int k, cudaDataType_t dt = CUDA_R_32F) {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, ld;
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, m, k));
  CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, n, k));
  CB(cublasLtMatrixLayoutCreate(&ld, dt, m, n, m));
  const size_t ws = (size_t)32 << 20;
  if (g_ws.ensure(ws)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
  cublasLtMatmulPreference_t pref;
  CB(cublasLtMatmulPreferenceCreate(&pref));
  CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t h; int nres = 0;
  CB(cublasLtMatmulAlgoGetHeuristic(g_lt, op, la, lb, ld, ld, pref, 1, &h, &nres));
  if (!nres) { snprintf(g_err, sizeof g_err, "no cuBLASLt FP8 algorithm for %dx%dx%d", m, n, k); return -1; }
  const float one = 1.f, zero = 0.f;
  CB(cublasLtMatmul(g_lt, op, &one, A, la, B, lb, &zero, D, ld, D, ld, &h.algo, g_ws.p, ws, g_stream));
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
  return 0;
}

static int gemm_f32_scores(const void* X, int64_t c0, int cols, int d, int nq) {
  const float one = 1.f, zero = 0.f;
  switch (g_var) {
    case V_F16:
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &one, (const __half*)X + c0 * d, CUDA_R_16F, d,
                      (const __half*)g_q.p, CUDA_R_16F, d, &zero, g_scores.p, CUDA_R_32F, cols, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
      return 0;
    case V_BF16:
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &one, (const __nv_bfloat16*)X + c0 * d, CUDA_R_16BF, d,
                      (const __nv_bfloat16*)g_qv.p, CUDA_R_16BF, d, &zero, g_scores.p, CUDA_R_32F, cols, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
      return 0;
    case V_F16ACC: {
      const __half h1 = __float2half(1.f), h0 = __float2half(0.f);
      if (g_tmp16.ensure((size_t)nq * cols * 2)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &h1, (const __half*)X + c0 * d, CUDA_R_16F, d,
                      (const __half*)g_q.p, CUDA_R_16F, d, &h0, g_tmp16.p, CUDA_R_16F, cols, CUBLAS_COMPUTE_16F,
                      CUBLAS_GEMM_DEFAULT));
      half_to_f32<<<(unsigned)(((int64_t)nq * cols + 255) / 256), 256, 0, g_stream>>>((const __half*)g_tmp16.p, (float*)g_scores.p, (int64_t)nq * cols);
      return 0;
    }
    case V_INT8: case V_INT8G: {
      if (nq <= 8 && d % 16 == 0) {   // small batches: memory-bound dp4a kernel (cuBLAS int8 GEMV is slow here)
        int8_gemv<<<(unsigned)((cols + 7) / 8), 256, (size_t)nq * d, g_stream>>>((const int8_t*)X, c0, cols, d, (const int8_t*)g_qv.p, nq,
                                                                      (const float*)g_xscale.p, (const float*)g_qscale.p, (float*)g_scores.p);
        return 0;
      }
      const int32_t i1 = 1, i0 = 0;
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &i1, (const int8_t*)X + c0 * d, CUDA_R_8I, d,
                      (const int8_t*)g_qv.p, CUDA_R_8I, d, &i0, g_scores.p, CUDA_R_32I, cols, CUBLAS_COMPUTE_32I,
                      CUBLAS_GEMM_DEFAULT));
      apply_scales<<<(unsigned)(((int64_t)nq * cols + 255) / 256), 256, 0, g_stream>>>((float*)g_scores.p, nq, cols, c0, 1,
                                                                                     (const float*)g_qscale.p, (const float*)g_xscale.p);
      return 0;
    }
    case V_FP8:
      if (lt_fp8((const uint8_t*)X + c0 * d, g_qv.p, (float*)g_scores.p, cols, nq, d)) return -1;
      apply_scales<<<(unsigned)(((int64_t)nq * cols + 255) / 256), 256, 0, g_stream>>>((float*)g_scores.p, nq, cols, c0, 0,
                                                                                     (const float*)g_qscale.p, (const float*)g_xscale.p);
      return 0;
  }
  snprintf(g_err, sizeof g_err, "unknown variant %d", g_var);
  return -1;
}


// Large-batch scores in 2 bytes (f16 / bf16) or int32 (int8), written to g_s16 (row-major nq x cols).
static int gemm_s16(const void* X, int64_t c0, int cols, int d, int nq) {
  const float one = 1.f, zero = 0.f;
  g_ld16 = cols;
  switch (g_var) {
    case V_F16:
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &one, (const __half*)X + c0 * d, CUDA_R_16F, d,
                      (const __half*)g_q.p, CUDA_R_16F, d, &zero, g_s16.p, CUDA_R_16F, cols, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
      return 0;
    case V_BF16:
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &one, (const __nv_bfloat16*)X + c0 * d, CUDA_R_16BF, d,
                      (const __nv_bfloat16*)g_qv.p, CUDA_R_16BF, d, &zero, g_s16.p, CUDA_R_32F, cols, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
      return 0;
    case V_F16ACC: {
      const __half h1 = __float2half(1.f), h0 = __float2half(0.f);
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &h1, (const __half*)X + c0 * d, CUDA_R_16F, d,
                      (const __half*)g_q.p, CUDA_R_16F, d, &h0, g_s16.p, CUDA_R_16F, cols, CUBLAS_COMPUTE_16F,
                      CUBLAS_GEMM_DEFAULT));
      return 0;
    }
    case V_INT8G:
#ifdef FC_HAVE_CUTLASS
      if (cutlass_ok()) {
        int m8 = (cols + 7) / 8 * 8;
        g_ld16 = m8;
        int rc = cutlass_kind() == 1 ? fc_cutlass_int8_gemm((const int8_t*)X + c0 * d, g_qv.p, g_s16.p, m8, nq, d, G_ALPHA, g_stream)
                                     : fc_cutlass80_int8_gemm((const int8_t*)X + c0 * d, g_qv.p, g_s16.p, m8, nq, d, G_ALPHA, g_stream);
        if (rc) { snprintf(g_err, sizeof g_err, "CUTLASS int8 GEMM failed (%d)", rc); return -1; }
        return 0;
      }
#endif
      [[fallthrough]];
    case V_INT8: {
      const int32_t i1 = 1, i0 = 0;
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &i1, (const int8_t*)X + c0 * d, CUDA_R_8I, d,
                      (const int8_t*)g_qv.p, CUDA_R_8I, d, &i0, g_s16.p, CUDA_R_32I, cols, CUBLAS_COMPUTE_32I,
                      CUBLAS_GEMM_DEFAULT));
      return 0;
    }
    case V_FP8:
      return lt_fp8((const uint8_t*)X + c0 * d, g_qv.p, g_s16.p, cols, nq, d, CUDA_R_32F);
  }
  return -1;
}

static int prep_queries(int nq, int d) {
  int64_t n = (int64_t)nq * d;
  f32_to_f16<<<(unsigned)((n + 255) / 256), 256, 0, g_stream>>>((const float*)g_qf.p, (__half*)g_q.p, n);
  if (g_var == V_BF16) {
    if (g_qv.ensure((size_t)n * 2)) return -1;
    f32_to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, g_stream>>>((const float*)g_qf.p, (__nv_bfloat16*)g_qv.p, n);
  } else if (g_var == V_INT8 || g_var == V_FP8 || g_var == V_INT8G) {
    if (g_qv.ensure((size_t)n) || g_qscale.ensure((size_t)nq * 4) || g_qs2.ensure((size_t)nq * 4)) return -1;
    quantize_rows<<<(unsigned)((nq + 7) / 8), 256, 0, g_stream>>>((const __half*)g_q.p, nq, d, g_var == V_INT8G ? V_INT8 : g_var, g_qv.p, (float*)g_qscale.p);
    if (g_var == V_INT8G)   // CUTLASS f16 scores are acc * G_ALPHA: true score = f16 * qscale * sg / G_ALPHA
      scale_f32<<<(unsigned)((nq + 255) / 256), 256, 0, g_stream>>>((const float*)g_qscale.p, (float*)g_qs2.p, nq, g_sg / G_ALPHA);
  }
  CK(cudaGetLastError());
  return 0;
}

// FLASHCLUSTER_PROFILE=1: per-phase device time (cudaEvents; adds syncs) printed to stderr after each search.
struct Prof {
  bool on = getenv("FLASHCLUSTER_PROFILE") != nullptr;
  cudaEvent_t a, b;
  double ms[4] = {0, 0, 0, 0};  // gemm, segment sort, filter, merge (+rescore)
  int chunks = 0, fallbacks = 0;
  Prof() { if (on) { cudaEventCreate(&a); cudaEventCreate(&b); } }
  void start() { if (on) cudaEventRecord(a, g_stream); }
  void stop(int i) {
    if (!on) return;
    cudaEventRecord(b, g_stream);
    cudaEventSynchronize(b);
    float t;
    cudaEventElapsedTime(&t, a, b);
    ms[i] += t;
  }
};

constexpr int64_t FIRST = (int64_t)SEG * 4;   // 16384 columns
// Score mode by batch size: below HALF_MIN_NQ queries score traffic is small, so f32 scores with a direct filter
// (no rescoring) and a 64k-column first chunk are fastest; from HALF_MIN_NQ on, f16 scores + exact rescoring halve the
// bytes written and read per chunk (bandwidth-bound regime). FLASHCLUSTER_HALF_MIN_NQ overrides.
static int half_min_nq() {   // measured: 128 on RTX PRO 6000 (sm_120), 512 on B200 (sm_100)
  static int v = [] {
    const char* e = getenv("FLASHCLUSTER_HALF_MIN_NQ");
    if (e) return atoi(e);
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major == 12 ? 128 : 512;
  }();
  return v;
}
static bool use_half(int nq) {   // int8g on CUTLASS: the large-batch path already wins from 128 queries
  return nq >= ((g_var == V_INT8G && cutlass_ok()) ? std::min(128, half_min_nq()) : half_min_nq());
}
// fused tensor-core score+filter kernel for int8 / f16 large batches (FLASHCLUSTER_FUSED=0: cuBLAS + filter)
static bool use_fused() {
  static int v = [] { const char* e = getenv("FLASHCLUSTER_FUSED"); return e ? atoi(e) : 0; }();   // experimental: WMMA is slower than cuBLAS here
  return v && (g_var == V_INT8 || g_var == V_F16) ;
}
static int s16_esz() {
  if (g_var == V_INT8G) return cutlass_ok() ? 2 : 4;
  return (g_var == V_INT8 || g_var == V_BF16 || g_var == V_FP8) ? 4 : 2;
}   // bytes per large-batch score
static int64_t l2_budget() {   // bytes of scores per chunk in the large-batch path (kept L2-resident)
  static int64_t v = [] { const char* e = getenv("FLASHCLUSTER_L2_MB"); return (int64_t)(e ? atoi(e) : 256) << 20; }();   // sweep: 24-96 MiB L2-sized chunks were not faster
  return v;
}
static int64_t chunk_cap(int nq) {   // columns per filter chunk: 256 MiB of f16 or of f32 scores
  int64_t budget = use_half(nq) ? l2_budget() : ((int64_t)1 << 28);
  int64_t c = (budget / ((int64_t)nq * (use_half(nq) ? s16_esz() : 4))) / 4096 * 4096;
  return std::max<int64_t>(4096, std::min<int64_t>((int64_t)1 << 21, c));
}
static int64_t first_cols(int nq) { return use_half(nq) ? FIRST : std::min<int64_t>((int64_t)SEG * CHUNK_SEGS, chunk_cap(nq)); }

static int seg_pass(const void* X, int64_t c0, int cols, int d, int nq, int k, Prof& pf, int64_t piece = FIRST) {
  const float one = 1.f, zero = 0.f;
  for (int64_t s0 = 0; s0 < cols; s0 += piece) {
    int sc = (int)std::min<int64_t>(piece, cols - s0);
    pf.start();
    if (gemm_f32_scores(X, c0 + s0, sc, d, nq)) return -1;
    pf.stop(0);
    int nseg = (sc + SEG - 1) / SEG;
    pf.start();
    seg_topk<<<dim3(nq, nseg), SORT_THREADS, 0, g_stream>>>((const float*)g_scores.p, sc, sc, c0 + s0, k,
                                                             (float*)g_cs.p, (int32_t*)g_ci.p);
    pf.stop(1);
    pf.start();
    merge_topk<<<nq, SORT_THREADS, 0, g_stream>>>((const float*)g_cs.p, (const int32_t*)g_ci.p, nullptr, nseg * k,
                                                  nseg * k, k, (float*)g_rs.p, (int32_t*)g_ri.p, (float*)g_tau.p, nullptr);
    pf.stop(3);
    CK(cudaGetLastError());
  }
  return 0;
}

static int exact_search_impl(const void* X, int64_t n, int d, int nq, int k, bool safe, Prof& pf) {
  const float one = 1.f, zero = 0.f;
  const bool half = use_half(nq);
  const int64_t cap = chunk_cap(nq), first = std::min<int64_t>(first_cols(nq), n);
  // f32 pieces for exact segment passes: at most CHUNK_SEGS segments, so one merge holds k + CHUNK_SEGS * k keys
  const int64_t piece = half ? FIRST : std::min<int64_t>(cap, (int64_t)SEG * CHUNK_SEGS);
  pf.chunks++;
  if (seg_pass(X, 0, (int)first, d, nq, k, pf, piece)) return -1;
  int64_t seen = first;
  for (int64_t c0 = first; c0 < n;) {
    int cols = (int)std::min<int64_t>(std::min(cap, std::max<int64_t>(first, seen * 8)), n - c0);
    pf.chunks++;
    if (safe) {
      if (seg_pass(X, c0, cols, d, nq, k, pf, piece)) return -1;
    } else {
      pf.start();
      const bool fused = half && use_fused();
      if (fused) {
        dim3 grid((nq + FBN - 1) / FBN, (cols + FBM - 1) / FBM);
        if (g_var == V_INT8)
          fused_score_filter<signed char, int><<<grid, 256, 0, g_stream>>>((const signed char*)X, (const signed char*)g_qv.p, c0, cols, nq, d,
              (const float*)g_xscale.p, (const float*)g_qscale.p, (const float*)g_tau.p, CAND, (float*)g_cs.p, (int32_t*)g_ci.p,
              (int*)g_cnt2.p, (int*)g_flag.p);
        else
          fused_score_filter<__half, float><<<grid, 256, 0, g_stream>>>((const __half*)X, (const __half*)g_q.p, c0, cols, nq, d,
              nullptr, nullptr, (const float*)g_tau.p, CAND, (float*)g_cs.p, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p);
        CK(cudaGetLastError());
      } else if (half) { if (gemm_s16(X, c0, cols, d, nq)) return -1; }
      else if (gemm_f32_scores(X, c0, cols, d, nq)) return -1;
      pf.stop(0);
      pf.start();
      if (fused) {
      } else if (half && g_var == V_F16) {
        int gy = (int)std::min<int64_t>(64, (cols + 2047) / 2048);
        filter_above16<<<dim3(nq, gy), 256, 0, g_stream>>>((const __half*)g_s16.p, cols, cols, c0, (const float*)g_tau.p,
                                                           CAND, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p);
      } else if (half) {
        int gy = (int)std::min<int64_t>(64, (cols + 1023) / 1024);
        dim3 grid(nq, gy);
        const float* qs = (g_var == V_INT8 || g_var == V_FP8 || g_var == V_INT8G) ? (const float*)g_qscale.p : nullptr;
        const float* xs = qs ? (const float*)g_xscale.p : nullptr;
        if (g_var == V_INT8G && cutlass_ok())
          filter16_scaled<<<dim3(nq, (int)std::min<int64_t>(64, (cols + 2047) / 2048)), 256, 0, g_stream>>>((const __half*)g_s16.p, g_ld16, cols, c0,
              (const float*)g_tau.p, (const float*)g_qs2.p, CAND, (float*)g_cs.p, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p);
        else if (g_var == V_INT8 || g_var == V_INT8G)
          filter_var<int32_t><<<grid, 256, 0, g_stream>>>((const int32_t*)g_s16.p, cols, cols, c0, (const float*)g_tau.p, CAND,
                                                          (float*)g_cs.p, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p, qs, xs);
        else if (g_var == V_BF16 || g_var == V_FP8)
          filter_var<float><<<grid, 256, 0, g_stream>>>((const float*)g_s16.p, cols, cols, c0, (const float*)g_tau.p, CAND,
                                                        (float*)g_cs.p, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p, qs, xs);
        else
          filter_var<__half><<<grid, 256, 0, g_stream>>>((const __half*)g_s16.p, cols, cols, c0, (const float*)g_tau.p, CAND,
                                                         (float*)g_cs.p, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p, qs, xs);
      } else {
        int gy = (int)std::min<int64_t>(64, (cols + 1023) / 1024);
        filter_above<<<dim3(nq, gy), 256, 0, g_stream>>>((const float*)g_scores.p, cols, cols, c0, (const float*)g_tau.p,
                                                         CAND, (float*)g_cs.p, (int32_t*)g_ci.p, (int*)g_cnt2.p, (int*)g_flag.p);
      }
      if (half) {   // large batch: no host sync; merge picks its sort size per row; overflow checked once at the end
        pf.stop(2);
        pf.start();
        if (g_var == V_F16 && !fused)
          rescore<<<nq, 256, 0, g_stream>>>((const __half*)X, (const __half*)g_q.p, d, (const int32_t*)g_ci.p,
                                            (const int*)g_cnt2.p, CAND, (float*)g_cs.p);
        merge_adaptive<<<nq, 256, 0, g_stream>>>((const float*)g_cs.p, (const int32_t*)g_ci.p, (int*)g_cnt2.p, CAND, k,
                                                 (float*)g_rs.p, (int32_t*)g_ri.p, (float*)g_tau.p);
        pf.stop(3);
        CK(cudaGetLastError());
        c0 += cols;
        seen += cols;
        continue;
      }
      int stat[2] = {0, 0};
      CK(cudaMemcpyAsync(stat, g_flag.p, 8, cudaMemcpyDeviceToHost, g_stream));
      CK(cudaStreamSynchronize(g_stream));
      pf.stop(2);
      pf.start();
      if (!stat[0]) {
        if (stat[1] > 0) {
          if (half && g_var == V_F16 && !fused)
            rescore<<<nq, 256, 0, g_stream>>>((const __half*)X, (const __half*)g_q.p, d, (const int32_t*)g_ci.p,
                                              (const int*)g_cnt2.p, CAND, (float*)g_cs.p);
          if (stat[1] + k <= SMALL_N)
            merge_topk_small<<<nq, 128, 0, g_stream>>>((const float*)g_cs.p, (const int32_t*)g_ci.p, (const int*)g_cnt2.p, 0,
                                                       CAND, k, (float*)g_rs.p, (int32_t*)g_ri.p, (float*)g_tau.p, (int*)g_cnt2.p);
          else
            merge_topk<<<nq, SORT_THREADS, 0, g_stream>>>((const float*)g_cs.p, (const int32_t*)g_ci.p, (const int*)g_cnt2.p, 0,
                                                          CAND, k, (float*)g_rs.p, (int32_t*)g_ri.p, (float*)g_tau.p, (int*)g_cnt2.p);
        }
        CK(cudaMemsetAsync(g_flag.p, 0, 8, g_stream));
      } else {  // a row overflowed: redo this chunk exactly
        pf.fallbacks++;
        CK(cudaMemsetAsync(g_cnt2.p, 0, (size_t)nq * 4, g_stream));
        CK(cudaMemsetAsync(g_flag.p, 0, 8, g_stream));
        if (seg_pass(X, c0, cols, d, nq, k, pf, piece)) return -1;
      }
      pf.stop(3);
    }
    CK(cudaGetLastError());
    c0 += cols;
    seen += cols;
  }
  return 0;
}

// Search into the device buffers g_rs / g_ri (nq x k, best first after the last merge).
static int search_dev(const void* X, int64_t n, int d, const float* q, int nq, int k) {
  if (k > SORT_N - CAND || k > SMALL_N / 2) { snprintf(g_err, sizeof g_err, "k too large"); return -1; }
  size_t cand = (size_t)nq * std::max<int64_t>(CAND, CHUNK_SEGS * k);
  if (g_q.ensure((size_t)nq * d * 2) || g_qf.ensure((size_t)nq * d * 4) || g_scores.ensure((size_t)nq * (use_half(nq) ? FIRST : chunk_cap(nq)) * 4) ||
      (use_half(nq) && !use_fused() && g_s16.ensure((size_t)nq * chunk_cap(nq) * s16_esz())) || g_rs.ensure((size_t)nq * k * 4) || g_ri.ensure((size_t)nq * k * 4) ||
      g_cs.ensure(cand * 4) || g_ci.ensure(cand * 4) || g_tau.ensure((size_t)nq * 4) || g_cnt2.ensure((size_t)nq * 4) ||
      g_flag.ensure(8)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
  CK(cudaMemcpyAsync(g_qf.p, q, (size_t)nq * d * 4, cudaMemcpyHostToDevice, g_stream));
  if (prep_queries(nq, d)) return -1;
  Prof pf;
  std::vector<float> init_s((size_t)nq * k, -INFINITY);
  std::vector<int32_t> init_i((size_t)nq * k, INT32_MAX);
  CK(cudaMemcpyAsync(g_rs.p, init_s.data(), init_s.size() * 4, cudaMemcpyHostToDevice, g_stream));
  CK(cudaMemcpyAsync(g_ri.p, init_i.data(), init_i.size() * 4, cudaMemcpyHostToDevice, g_stream));
  CK(cudaMemsetAsync(g_cnt2.p, 0, (size_t)nq * 4, g_stream));
  CK(cudaMemsetAsync(g_flag.p, 0, 8, g_stream));
  if (exact_search_impl(X, n, d, nq, k, false, pf)) return -1;
  if (use_half(nq)) {   // the large-batch path does not stop per chunk: if any row overflowed, redo exactly
    int stat0 = 0;
    CK(cudaMemcpyAsync(&stat0, g_flag.p, 4, cudaMemcpyDeviceToHost, g_stream));
    CK(cudaStreamSynchronize(g_stream));
    if (stat0) {
      pf.fallbacks++;
      CK(cudaMemcpyAsync(g_rs.p, init_s.data(), init_s.size() * 4, cudaMemcpyHostToDevice, g_stream));
      CK(cudaMemcpyAsync(g_ri.p, init_i.data(), init_i.size() * 4, cudaMemcpyHostToDevice, g_stream));
      CK(cudaMemsetAsync(g_cnt2.p, 0, (size_t)nq * 4, g_stream));
      CK(cudaMemsetAsync(g_flag.p, 0, 8, g_stream));
      if (exact_search_impl(X, n, d, nq, k, true, pf)) return -1;
    }
  }
  if (pf.on) {
    CK(cudaStreamSynchronize(g_stream));
    fprintf(stderr, "{\"profile\":{\"nq\":%d,\"chunks\":%d,\"fallback_chunks\":%d,\"gemm_ms\":%.3f,\"seg_ms\":%.3f,\"filter_ms\":%.3f,\"merge_ms\":%.3f}}\n",
            nq, pf.chunks, pf.fallbacks, pf.ms[0], pf.ms[1], pf.ms[2], pf.ms[3]);
  }
  return 0;
}

int fc_exact_search(const void* X, int64_t n, int d, const float* q, int nq, int k, float* out_s, int32_t* out_i) {
  if (search_dev(X, n, d, q, nq, k)) return -1;
  CK(cudaMemcpyAsync(out_s, g_rs.p, (size_t)nq * k * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaMemcpyAsync(out_i, g_ri.p, (size_t)nq * k * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

// Pin and map the host f16 corpus (n x d) for exact re-ranking from the device without a device copy.
const __half* g_hostdev = nullptr;
int fc_register_host(const uint16_t* host, int64_t n, int d) {
  CK(cudaHostRegister((void*)host, (size_t)n * d * 2, cudaHostRegisterMapped));
  void* dp = nullptr;
  CK(cudaHostGetDevicePointer(&dp, (void*)host, 0));
  g_hostdev = (const __half*)dp;
  return 0;
}

// Speed mode for re-ranking: a device-resident f16 copy (more device memory, no PCIe reads).
int fc_rerank_device_copy(const uint16_t* host, int64_t n, int d) {
  void* p = nullptr;
  CK(cudaMalloc(&p, (size_t)n * d * 2));
  CK(cudaMemcpy(p, host, (size_t)n * d * 2, cudaMemcpyHostToDevice));
  g_hostdev = (const __half*)p;
  return 0;
}

// Approximate search for a shortlist of L, then exact f16 x f16 (f32 accumulate) re-rank on the device from the
// mapped host rows; returns the best `depth`.
int fc_search_rerank(const void* X, int64_t n, int d, const float* q, int nq, int L, int depth, float* out_s, int32_t* out_i) {
  if (!g_hostdev) { snprintf(g_err, sizeof g_err, "fc_register_host not called"); return -1; }
  if (depth + L > SMALL_N) { snprintf(g_err, sizeof g_err, "depth + shortlist must be <= %d", SMALL_N); return -1; }
  if (search_dev(X, n, d, q, nq, L)) return -1;
  if (g_pc.ensure((size_t)nq * L * 4) || g_w.ensure((size_t)nq * depth * 8) || g_pairs.ensure((size_t)nq * depth * 4) ||
      g_tmp16.ensure((size_t)nq * 4)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
  float* rs = (float*)g_w.p;            // nq x depth running scores
  int32_t* ri = (int32_t*)g_pairs.p;    // nq x depth running ids
  rerank_host<<<nq, 256, 0, g_stream>>>(g_hostdev, (const __half*)g_q.p, d, (const int32_t*)g_ri.p, L, (float*)g_pc.p);
  std::vector<float> init_s((size_t)nq * depth, -INFINITY);
  std::vector<int32_t> init_i((size_t)nq * depth, INT32_MAX);
  CK(cudaMemcpyAsync(rs, init_s.data(), init_s.size() * 4, cudaMemcpyHostToDevice, g_stream));
  CK(cudaMemcpyAsync(ri, init_i.data(), init_i.size() * 4, cudaMemcpyHostToDevice, g_stream));
  merge_topk_small<<<nq, 128, 0, g_stream>>>((const float*)g_pc.p, (const int32_t*)g_ri.p, nullptr, L, L, depth, rs, ri,
                                             (float*)g_tmp16.p, nullptr);
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(out_s, rs, (size_t)nq * depth * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaMemcpyAsync(out_i, ri, (size_t)nq * depth * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

// Batch statistics for rows `rows` (m of them; null = rows 0..m) of device f16 matrix X against host f32 centroids
// (k x d). Outputs (host): labels (m), dist_sum, sums (k x d), counts (k).
int fc_batch_stats(const void* X, int d, const int32_t* rows, int64_t m, const float* cent, int k,
                    int32_t* out_lab, double* out_dist, float* out_sums, uint32_t* out_cnt) {
  if (g_rows.ensure((size_t)m * 4) || g_xb.ensure((size_t)m * d * 4) || g_s.ensure((size_t)m * k * 4) ||
      g_c.ensure((size_t)k * d * 4) || g_c2.ensure((size_t)k * 4) || g_x2.ensure((size_t)m * 4) ||
      g_lab.ensure((size_t)m * 4) || g_dist.ensure((size_t)m * 4) || g_sums.ensure((size_t)k * d * 4) ||
      g_cnt.ensure((size_t)k * 4)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
  if (rows) CK(cudaMemcpyAsync(g_rows.p, rows, (size_t)m * 4, cudaMemcpyHostToDevice, g_stream));
  CK(cudaMemcpyAsync(g_c.p, cent, (size_t)k * d * 4, cudaMemcpyHostToDevice, g_stream));
  int64_t tot = m * d;
  gather_f16_to_f32<<<(unsigned)((tot + 255) / 256), 256, 0, g_stream>>>((const __half*)X, rows ? (const int32_t*)g_rows.p : nullptr, m, d, (float*)g_xb.p);
  row_sqnorm<<<(unsigned)((k + 7) / 8), 256, 0, g_stream>>>((const float*)g_c.p, k, d, (float*)g_c2.p);
  row_sqnorm<<<(unsigned)((m + 7) / 8), 256, 0, g_stream>>>((const float*)g_xb.p, m, d, (float*)g_x2.p);
  const float one = 1.f, zero = 0.f;
  // S^T (k x m) col-major = C (k x d) . XB^T  => row-major S (m x k)
  CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, k, (int)m, d, &one, (const float*)g_c.p, d, (const float*)g_xb.p, d,
                 &zero, (float*)g_s.p, k));
  argmin_l2<<<(unsigned)((m + 7) / 8), 256, 0, g_stream>>>((const float*)g_s.p, m, k, (const float*)g_c2.p,
                                                         (const float*)g_x2.p, (int32_t*)g_lab.p, (float*)g_dist.p);
  if (out_sums) {
    CK(cudaMemsetAsync(g_sums.p, 0, (size_t)k * d * 4, g_stream));
    CK(cudaMemsetAsync(g_cnt.p, 0, (size_t)k * 4, g_stream));
    accumulate<<<(unsigned)((tot + 255) / 256), 256, 0, g_stream>>>((const float*)g_xb.p, m, d, (const int32_t*)g_lab.p,
                                                                     (float*)g_sums.p, (uint32_t*)g_cnt.p);
  }
  CK(cudaGetLastError());
  std::vector<float> dist(m);
  CK(cudaMemcpyAsync(out_lab, g_lab.p, (size_t)m * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaMemcpyAsync(dist.data(), g_dist.p, (size_t)m * 4, cudaMemcpyDeviceToHost, g_stream));
  if (out_sums) {
    CK(cudaMemcpyAsync(out_sums, g_sums.p, (size_t)k * d * 4, cudaMemcpyDeviceToHost, g_stream));
    CK(cudaMemcpyAsync(out_cnt, g_cnt.p, (size_t)k * 4, cudaMemcpyDeviceToHost, g_stream));
  }
  CK(cudaStreamSynchronize(g_stream));
  double s = 0;
  for (int64_t i = 0; i < m; ++i) s += dist[i];
  *out_dist = s;
  return 0;
}

// ---------------------------------------------------------------- device-resident e-means (see kmeans::EmeansEngine)
int fc_em_init(const float* c, int k, int d, int mode) {
  g_em_mode = mode;
  if (g_C.ensure((size_t)k * d * 4) || g_close.ensure((size_t)k * d * 4) || g_w.ensure((size_t)k * 4) ||
      g_pc.ensure((size_t)k * 8) || g_sums.ensure((size_t)k * d * 4) || g_cnt.ensure((size_t)k * 4) ||
      g_c2.ensure((size_t)k * 4)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
  g_k = k; g_d = d;
  CK(cudaMemcpyAsync(g_C.p, c, (size_t)k * d * 4, cudaMemcpyHostToDevice, g_stream));
  CK(cudaMemsetAsync(g_close.p, 0, (size_t)k * d * 4, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

// Assign rows to the device centroids; keep per-cluster sums on the device (and add them to the closing sums when
// `closing`). Returns per-cluster counts and the sum of squared distances.
int fc_em_batch(const void* X, const int32_t* rows, int64_t m, int closing, uint32_t* out_cnt, double* out_dist) {
  const int k = g_k, d = g_d;
  if (g_rows.ensure((size_t)m * 4) || g_xb.ensure((size_t)m * d * 4) || g_s.ensure((size_t)m * k * 4) ||
      g_x2.ensure((size_t)m * 4) || g_lab.ensure((size_t)m * 4) || g_dist.ensure((size_t)m * 4)) {
    snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1;
  }
  CK(cudaMemcpyAsync(g_rows.p, rows, (size_t)m * 4, cudaMemcpyHostToDevice, g_stream));
  int64_t tot = m * d;
  gather_f16_to_f32<<<(unsigned)((tot + 255) / 256), 256, 0, g_stream>>>((const __half*)X, (const int32_t*)g_rows.p, m, d, (float*)g_xb.p);
  row_sqnorm<<<(unsigned)((k + 7) / 8), 256, 0, g_stream>>>((const float*)g_C.p, k, d, (float*)g_c2.p);
  row_sqnorm<<<(unsigned)((m + 7) / 8), 256, 0, g_stream>>>((const float*)g_xb.p, m, d, (float*)g_x2.p);
  const float one = 1.f, zero = 0.f;
  if (g_em_mode <= 1) {   // f32 SGEMM, or TF32 tensor cores for this call only
    if (g_em_mode == 1) CB(cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH));
    CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, k, (int)m, d, &one, (const float*)g_C.p, d, (const float*)g_xb.p, d,
                   &zero, (float*)g_s.p, k));
    if (g_em_mode == 1) CB(cublasSetMathMode(g_blas, CUBLAS_DEFAULT_MATH));
  } else if (g_em_mode >= 4) {   // int8 / fp8 inputs with per-row scales (centroids re-quantised each step)
    const int var = g_em_mode == 4 ? V_INT8 : V_FP8;
    if (g_c16.ensure((size_t)k * d * 2) || g_xb16.ensure((size_t)m * d * 2) || g_c8.ensure((size_t)k * d) ||
        g_x8.ensure((size_t)m * d) || g_cs8.ensure((size_t)k * 4) || g_xs8.ensure((size_t)m * 4)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
    f32_to_f16<<<(unsigned)(((int64_t)k * d + 255) / 256), 256, 0, g_stream>>>((const float*)g_C.p, (__half*)g_c16.p, (int64_t)k * d);
    quantize_rows<<<(unsigned)((k + 7) / 8), 256, 0, g_stream>>>((const __half*)g_c16.p, k, d, var, g_c8.p, (float*)g_cs8.p);
    gather_f16<<<(unsigned)((tot + 255) / 256), 256, 0, g_stream>>>((const __half*)X, (const int32_t*)g_rows.p, m, d, (__half*)g_xb16.p, 0);
    quantize_rows<<<(unsigned)((m + 7) / 8), 256, 0, g_stream>>>((const __half*)g_xb16.p, m, d, var, g_x8.p, (float*)g_xs8.p);
    if (var == V_INT8) {
      const int32_t i1 = 1, i0 = 0;
      CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, k, (int)m, d, &i1, g_c8.p, CUDA_R_8I, d, g_x8.p, CUDA_R_8I, d, &i0,
                      g_s.p, CUDA_R_32I, k, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    } else if (lt_fp8(g_c8.p, g_x8.p, g_s.p, k, (int)m, d)) return -1;
    argmin_l2_q<<<(unsigned)((m + 7) / 8), 256, 0, g_stream>>>(g_s.p, var == V_INT8, m, k, (const float*)g_c2.p, (const float*)g_x2.p,
                                                               (const float*)g_cs8.p, (const float*)g_xs8.p, (int32_t*)g_lab.p, (float*)g_dist.p);
  } else {                // f16 / bf16 inputs (centroids rounded each step), f32 accumulate
    const bool bf = g_em_mode == 3;
    if (g_c16.ensure((size_t)k * d * 2) || g_xb16.ensure((size_t)m * d * 2)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
    if (bf) f32_to_bf16<<<(unsigned)(((int64_t)k * d + 255) / 256), 256, 0, g_stream>>>((const float*)g_C.p, (__nv_bfloat16*)g_c16.p, (int64_t)k * d);
    else f32_to_f16<<<(unsigned)(((int64_t)k * d + 255) / 256), 256, 0, g_stream>>>((const float*)g_C.p, (__half*)g_c16.p, (int64_t)k * d);
    gather_f16<<<(unsigned)((tot + 255) / 256), 256, 0, g_stream>>>((const __half*)X, (const int32_t*)g_rows.p, m, d, (__half*)g_xb16.p, bf);
    cudaDataType_t t = bf ? CUDA_R_16BF : CUDA_R_16F;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, k, (int)m, d, &one, g_c16.p, t, d, g_xb16.p, t, d, &zero, g_s.p,
                    CUDA_R_32F, k, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  if (g_em_mode < 4)
    argmin_l2<<<(unsigned)((m + 7) / 8), 256, 0, g_stream>>>((const float*)g_s.p, m, k, (const float*)g_c2.p,
                                                           (const float*)g_x2.p, (int32_t*)g_lab.p, (float*)g_dist.p);
  CK(cudaMemsetAsync(g_sums.p, 0, (size_t)k * d * 4, g_stream));
  CK(cudaMemsetAsync(g_cnt.p, 0, (size_t)k * 4, g_stream));
  accumulate<<<(unsigned)((tot + 255) / 256), 256, 0, g_stream>>>((const float*)g_xb.p, m, d, (const int32_t*)g_lab.p,
                                                                   (float*)g_sums.p, (uint32_t*)g_cnt.p);
  if (closing)
    add_inplace<<<(unsigned)(((int64_t)k * d + 255) / 256), 256, 0, g_stream>>>((float*)g_close.p, (const float*)g_sums.p, (int64_t)k * d);
  CK(cudaGetLastError());
  std::vector<float> dist(m);
  CK(cudaMemcpyAsync(out_cnt, g_cnt.p, (size_t)k * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaMemcpyAsync(dist.data(), g_dist.p, (size_t)m * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  double sd = 0;
  for (int64_t i = 0; i < m; ++i) sd += dist[i];
  *out_dist = sd;
  return 0;
}

int fc_em_update(const float* w) {
  const int k = g_k, d = g_d;
  CK(cudaMemcpyAsync(g_w.p, w, (size_t)k * 4, cudaMemcpyHostToDevice, g_stream));
  em_update<<<(unsigned)(((int64_t)k * d + 255) / 256), 256, 0, g_stream>>>((float*)g_C.p, (const float*)g_sums.p,
                                                                            (const uint32_t*)g_cnt.p, (const float*)g_w.p, k, d);
  CK(cudaGetLastError());
  return 0;
}

int fc_em_split(const int32_t* pairs, int np) {
  if (g_pairs.ensure((size_t)np * 8)) { snprintf(g_err, sizeof g_err, "cudaMalloc failed"); return -1; }
  CK(cudaMemcpyAsync(g_pairs.p, pairs, (size_t)np * 8, cudaMemcpyHostToDevice, g_stream));
  em_split<<<np, 256, 0, g_stream>>>((float*)g_C.p, (const int32_t*)g_pairs.p, g_d);
  CK(cudaGetLastError());
  return 0;
}

int fc_em_finalize(const double* pass_counts, float* out_c) {
  const int k = g_k, d = g_d;
  CK(cudaMemcpyAsync(g_pc.p, pass_counts, (size_t)k * 8, cudaMemcpyHostToDevice, g_stream));
  em_close<<<(unsigned)(((int64_t)k * d + 255) / 256), 256, 0, g_stream>>>((float*)g_C.p, (const float*)g_close.p,
                                                                           (const double*)g_pc.p, k, d);
  CK(cudaMemcpyAsync(out_c, g_C.p, (size_t)k * d * 4, cudaMemcpyDeviceToHost, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

}  // extern "C"
