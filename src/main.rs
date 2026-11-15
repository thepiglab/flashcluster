//! flashcluster command line.
//!   fit           cluster an .npy matrix (e-means or Lloyd), write centroids and labels
//!   prep          write shared benchmark inputs (f16 data, queries, centroids, exact ground truth)
//!   search-bench  CPU IVF: recall, latency, throughput, memory from a prep dir
//!   gpu-bench     GPU exact search and e-means from a prep dir (feature `cuda`)
use clap::{Parser, Subcommand};
use flashcluster::bench;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// worker threads (default: all available)
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Fit(bench::FitArgs),
    Prep(bench::PrepArgs),
    SearchBench(bench::SearchBenchArgs),
    #[cfg(feature = "cuda")]
    GpuBench(flashcluster::gpu::GpuBenchArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }
    match cli.cmd {
        Cmd::Fit(a) => bench::fit(a),
        Cmd::Prep(a) => bench::prep(a),
        Cmd::SearchBench(a) => bench::search_bench(a),
        #[cfg(feature = "cuda")]
        Cmd::GpuBench(a) => flashcluster::gpu::bench(a),
    }
}
