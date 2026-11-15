// CUTLASS (tcgen05, SM100) int8 x int8 -> int32 GEMM with an f16 output epilogue (D = alpha * acc), used for
// large-batch int8 search on B200: half the score bytes of cuBLAS's int32 output, at int8 tensor-core rate.
// Layouts (TN): A = corpus chunk, row-major M x K (M = corpus rows); B = queries, column-major K x N (N = queries);
// D column-major M x N, i.e. row-major [query][corpus row], ld = M, which is what the score filter reads.
// Built only when FLASHCLUSTER_CUTLASS points at CUTLASS's include dirs (see build.rs); compiled for sm_100a.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
namespace {
using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;
using ElementD = cutlass::half_t;
using LayoutD = cutlass::layout::ColumnMajor;
constexpr int AlignmentD = 8;
using ElementAcc = int32_t;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OpClass = cutlass::arch::OpClassTensorOp;
// 2-SM tcgen05 tile (each SM 128 x 256); int8 K-step 128. Override at build time with FC_CUTLASS_TILE_N=128.
#ifndef FC_CUTLASS_TILE_N
#define FC_CUTLASS_TILE_N 256
#endif
using MmaTile = Shape<_256, Int<FC_CUTLASS_TILE_N>, _128>;
using Cluster = Shape<_2, _1, _1>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass, MmaTile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    void, LayoutD, AlignmentD,              // no source C
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAcc, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

Gemm g_gemm;
void* g_ws = nullptr;
size_t g_ws_size = 0;
}  // namespace
#endif

extern "C" int fc_cutlass_int8_available() {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  return major == 10 && minor == 0;
#else
  return 0;
#endif
}

// D (f16, ld m) = alpha * A (m x k int8, row-major) . B^T (n x k int8, row-major == k x n column-major)
extern "C" int fc_cutlass_int8_gemm(const void* A, const void* B, void* D, int m, int n, int k, float alpha,
                                    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto sA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto sB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto sC = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {(const ElementA*)A, sA, (const ElementB*)B, sB},
      {{alpha, 0.f}, nullptr, sC, (ElementD*)D, sD}};
  cutlass::Status st = g_gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS can_implement m=%d n=%d k=%d: %s\n", m, n, k, cutlass::cutlassGetStatusString(st));
    return -2;
  }
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > g_ws_size) {
    if (g_ws) cudaFree(g_ws);
    if (cudaMalloc(&g_ws, ws) != cudaSuccess) { g_ws = nullptr; g_ws_size = 0; return -3; }
    g_ws_size = ws;
  }
  if (g_gemm.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return -4;
  if (g_gemm.run(stream) != cutlass::Status::kSuccess) return -5;
  return 0;
#else
  (void)A; (void)B; (void)D; (void)m; (void)n; (void)k; (void)alpha; (void)stream;
  return -1;
#endif
}
