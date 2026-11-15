// CUTLASS 2.x int8 tensor-core GEMM (mma.sync m16n8k32, Ampere-style; runs on SM80+, used on SM120 / RTX Blackwell,
// which has no tcgen05) with an f16 output epilogue: D = alpha * acc, no source read. Same layouts as the SM100 kernel:
// A corpus chunk row-major M x K, B queries column-major K x N, D column-major M x N (= row-major [query][row], ld = M).
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

namespace {
using Epi = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float,
                                                         cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
using Gemm80 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    Epi, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
Gemm80 g80;
}  // namespace

extern "C" int fc_cutlass80_int8_available() {
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  return major == 12;
}

extern "C" int fc_cutlass80_int8_gemm(const void* A, const void* B, void* D, int m, int n, int k, float alpha,
                                      cudaStream_t stream) {
  Gemm80::Arguments args({m, n, k}, {(const int8_t*)A, k}, {(const int8_t*)B, k}, {(cutlass::half_t*)D, m},
                         {(cutlass::half_t*)D, m}, {alpha});
  cutlass::Status st = g80.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS sm80 can_implement m=%d n=%d k=%d: %s\n", m, n, k, cutlass::cutlassGetStatusString(st));
    return -2;
  }
  if (g80.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return -4;
  if (g80.run(stream) != cutlass::Status::kSuccess) return -5;
  return 0;
}
