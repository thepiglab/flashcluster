// With feature `cuda`: compile cuda/flashcluster.cu with nvcc (CUDA_HOME or nvcc on PATH) and link cudart + cuBLAS.
// FLASHCLUSTER_CUDA_ARCH: comma-separated SM versions (default "100,120": B200 and RTX PRO 6000 Blackwell); PTX for the highest
// one is embedded too, so newer GPUs JIT it. Same scheme as flashembed's GPU runner.
fn main() {
    println!("cargo:rerun-if-changed=cuda/flashcluster.cu");
    println!("cargo:rerun-if-env-changed=FLASHCLUSTER_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=FLASHCLUSTER_SKIP_NVCC");
    // FLASHCLUSTER_SKIP_NVCC=1: type-check the `cuda` feature on a machine without CUDA (does not link)
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() || std::env::var_os("FLASHCLUSTER_SKIP_NVCC").is_some() {
        return;
    }
    let archs = std::env::var("FLASHCLUSTER_CUDA_ARCH").unwrap_or_else(|_| "100,120".into());
    let mut b = cc::Build::new();
    // debug(false): the release profile keeps host debug info, and cc would otherwise pass -G (device debug,
    // unoptimised kernels: ~100x slower block sorts)
    b.cuda(true).cudart("shared").debug(false).file("cuda/flashcluster.cu").flag("-O3").flag("-std=c++17").flag("-lineinfo").warnings(false);
    let list: Vec<&str> = archs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for a in &list {
        b.flag(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    if let Some(last) = list.last() {
        b.flag(format!("-gencode=arch=compute_{last},code=compute_{last}"));
    }
    // Optional CUTLASS kernels (FLASHCLUSTER_CUTLASS = CUTLASS source root, headers only): SM100 int8 GEMM with an
    // f16 epilogue, compiled for sm_100a in its own object; the main engine calls it on B200-class GPUs.
    println!("cargo:rerun-if-env-changed=FLASHCLUSTER_CUTLASS");
    println!("cargo:rerun-if-changed=cuda/cutlass_int8.cu");
    if let Ok(root) = std::env::var("FLASHCLUSTER_CUTLASS") {
        let root = std::path::PathBuf::from(root);
        let mut c = cc::Build::new();
        c.cuda(true).cudart("shared").debug(false).file("cuda/cutlass_int8.cu").flag("-O3").flag("-std=c++17")
            .flag("--expt-relaxed-constexpr").flag("-DNDEBUG").include(root.join("include"))
            .include(root.join("tools/util/include")).flag("-gencode=arch=compute_100a,code=sm_100a").warnings(false);
        c.compile("fc_cutlass");
        // SM120 (RTX Blackwell): CUTLASS 2.x int8 mma.sync GEMM, f16 epilogue
        println!("cargo:rerun-if-changed=cuda/cutlass_int8_sm80.cu");
        let mut c80 = cc::Build::new();
        c80.cuda(true).cudart("shared").debug(false).file("cuda/cutlass_int8_sm80.cu").flag("-O3").flag("-std=c++17")
            .flag("--expt-relaxed-constexpr").flag("-DNDEBUG").include(root.join("include"))
            .flag("-gencode=arch=compute_120,code=sm_120").warnings(false);
        c80.compile("fc_cutlass80");
        b.define("FC_HAVE_CUTLASS", None);
    }
    b.compile("flashcluster");
    if let Some(home) = std::env::var_os("CUDA_HOME").or_else(|| std::env::var_os("CUDA_PATH")) {
        let home = std::path::PathBuf::from(home);
        for d in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            println!("cargo:rustc-link-search=native={}", home.join(d).display());
        }
    }
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
