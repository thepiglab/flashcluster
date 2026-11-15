//! flashcluster: k-means / e-means clustering and nearest-neighbour search for embeddings.
//!
//! * `kmeans`: e-means (EMA mini-batch k-means with fitted batch/step laws; port of FAISS's reference) and
//!   FAISS-style Lloyd k-means, with shared quality measures (objective, balance, ARI).
//! * `index`: IVF over f16 lists with AVX2/AVX-512 kernels, list-major batch search, in-place build.
//! * `linalg` / `simd`: GEMM-based exact search, nearest centroid, runtime-dispatched SIMD kernels.
//! * `gpu` (feature `cuda`): exact search on f16 tensor cores with a device top-k, and e-means batch statistics.
//! * `bench`: shared benchmark inputs (`prep`), CPU search benchmark, `fit`.
pub mod bench;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod index;
pub mod io;
pub mod kmeans;
pub mod linalg;
pub mod quant;
pub mod simd;
