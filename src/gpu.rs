//! GPU engine (feature `cuda`): safe wrappers over cuda/flashcluster.cu and the `gpu-bench` subcommand.
//!
//! * `Gpu::exact_search`: brute-force top-k inner product on a device-resident f16 matrix (tensor-core GEMM, f32
//!   accumulation, device top-k). At a few million vectors this replaces IVF on a GPU: it is exact and still fast.
//! * `GpuStats`: e-means batch statistics on the device; the e-means update logic stays in `kmeans::emeans_with`.
use crate::io::{self, Mat};
use crate::bench::{metrics, Prep};
use crate::kmeans::{emeans_with, BatchLaw, EmeansEngine};
use anyhow::{bail, Result};
use half::f16;
use serde_json::json;
use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use std::time::Instant;

extern "C" {
    fn fc_last_error() -> *const c_char;
    fn fc_init(device: i32, tf32: i32) -> i32;
    fn fc_device_name(out: *mut c_char, len: i32) -> i32;
    fn fc_upload_f16(host: *const u16, n: i64, d: i32) -> *mut c_void;
    fn fc_free(p: *mut c_void);
    fn fc_mem_used(used: *mut i64) -> i32;
    fn fc_exact_search(x: *const c_void, n: i64, d: i32, q: *const f32, nq: i32, k: i32, out_s: *mut f32, out_i: *mut i32) -> i32;
    fn fc_em_init(c: *const f32, k: i32, d: i32, mode: i32) -> i32;
    fn fc_upload_variant(host: *const u16, n: i64, d: i32, var: i32) -> *mut c_void;
    fn fc_register_host(host: *const u16, n: i64, d: i32) -> i32;
    fn fc_rerank_device_copy(host: *const u16, n: i64, d: i32) -> i32;
    fn fc_search_rerank(x: *const c_void, n: i64, d: i32, q: *const f32, nq: i32, l: i32, depth: i32, out_s: *mut f32, out_i: *mut i32) -> i32;
    fn fc_em_batch(x: *const c_void, rows: *const i32, m: i64, closing: i32, out_cnt: *mut u32, out_dist: *mut f64) -> i32;
    fn fc_em_update(w: *const f32) -> i32;
    fn fc_em_split(pairs: *const i32, np: i32) -> i32;
    fn fc_em_finalize(pass_counts: *const f64, out_c: *mut f32) -> i32;
    fn fc_batch_stats(x: *const c_void, d: i32, rows: *const i32, m: i64, cent: *const f32, k: i32, out_lab: *mut i32,
                       out_dist: *mut f64, out_sums: *mut f32, out_cnt: *mut u32) -> i32;
}

fn check(rc: i32) -> Result<()> {
    if rc != 0 {
        bail!("cuda: {}", unsafe { CStr::from_ptr(fc_last_error()) }.to_string_lossy());
    }
    Ok(())
}

/// A device-resident f16 matrix (n×d) plus the host copy (for initial centroids).
pub struct Gpu<'a> {
    dev: *mut c_void,
    pub n: usize,
    pub d: usize,
    host: &'a [f16],
}

impl Drop for Gpu<'_> {
    fn drop(&mut self) {
        unsafe { fc_free(self.dev) }
    }
}

pub fn init(device: i32, tf32: bool) -> Result<String> {
    check(unsafe { fc_init(device, tf32 as i32) })?;
    let mut buf = [0 as c_char; 256];
    check(unsafe { fc_device_name(buf.as_mut_ptr(), 256) })?;
    Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
}

pub fn device_mem_used_gb() -> f64 {
    let mut u = 0i64;
    unsafe { fc_mem_used(&mut u) };
    u as f64 / 1e9
}

/// Corpus storage / compute precision for GPU search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    F16,    // f16 storage, f16 tensor cores, f32 accumulate (default; f16 or f32 score buffers by batch size)
    Bf16,   // bf16 storage and inputs, f32 accumulate
    F16Acc, // f16 storage, f16 accumulate (CUBLAS_COMPUTE_16F)
    Int8,   // int8 storage, per-vector f32 scale, int32 accumulate
    Fp8,    // FP8 e4m3 storage, per-vector f32 scale, f32 accumulate (cuBLASLt)
    Int8g,  // int8 storage, one corpus scale; CUTLASS int8 GEMM with f16 scores on SM100, cuBLAS elsewhere
}

impl std::str::FromStr for Precision {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "f16" => Precision::F16,
            "bf16" => Precision::Bf16,
            "f16acc" => Precision::F16Acc,
            "int8" => Precision::Int8,
            "fp8" => Precision::Fp8,
            "int8g" => Precision::Int8g,
            _ => bail!("unknown precision {s}"),
        })
    }
}

impl<'a> Gpu<'a> {
    pub fn upload_variant(host: &'a [f16], n: usize, d: usize, p: Precision) -> Result<Self> {
        let var = match p { Precision::F16 => 0, Precision::Bf16 => 1, Precision::F16Acc => 2, Precision::Int8 => 3, Precision::Fp8 => 4, Precision::Int8g => 5 };
        let dev = unsafe { fc_upload_variant(host.as_ptr() as *const u16, n as i64, d as i32, var) };
        if dev.is_null() {
            bail!("device upload ({p:?}) failed: {}", unsafe { CStr::from_ptr(fc_last_error()) }.to_string_lossy());
        }
        Ok(Gpu { dev, n, d, host })
    }

    pub fn upload(host: &'a [f16], n: usize, d: usize) -> Result<Self> {
        let dev = unsafe { fc_upload_f16(host.as_ptr() as *const u16, n as i64, d as i32) };
        if dev.is_null() {
            bail!("device upload of {n}×{d} f16 failed (out of memory?)");
        }
        Ok(Gpu { dev, n, d, host })
    }

    /// Top-k (score, row) per query, best first; equal scores ordered by row id.
    pub fn exact_search(&self, q: &Mat, k: usize) -> Result<Vec<Vec<(f32, u32)>>> {
        let (mut s, mut i) = (vec![0f32; q.n * k], vec![0i32; q.n * k]);
        check(unsafe { fc_exact_search(self.dev, self.n as i64, self.d as i32, q.data.as_ptr(), q.n as i32, k as i32, s.as_mut_ptr(), i.as_mut_ptr()) })?;
        Ok((0..q.n).map(|r| {
            let mut v: Vec<(f32, u32)> = (0..k).filter(|&t| i[r * k + t] != i32::MAX).map(|t| (s[r * k + t], i[r * k + t] as u32)).collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            v
        }).collect())
    }

    /// Pin and map the host f16 corpus so `search_rerank` can read shortlisted rows without a device copy.
    pub fn register_host(&self) -> Result<()> {
        check(unsafe { fc_register_host(self.host.as_ptr() as *const u16, self.n as i64, self.d as i32) })
    }

    /// Speed mode: keep an f16 copy on the device for re-ranking (more device memory, no PCIe reads).
    pub fn rerank_device_copy(&self) -> Result<()> {
        check(unsafe { fc_rerank_device_copy(self.host.as_ptr() as *const u16, self.n as i64, self.d as i32) })
    }

    /// Approximate search for a shortlist of `l`, exact f16 re-rank on the device; top `depth` per query.
    pub fn search_rerank(&self, q: &Mat, l: usize, depth: usize) -> Result<Vec<Vec<(f32, u32)>>> {
        let (mut s, mut i) = (vec![0f32; q.n * depth], vec![0i32; q.n * depth]);
        check(unsafe { fc_search_rerank(self.dev, self.n as i64, self.d as i32, q.data.as_ptr(), q.n as i32, l as i32, depth as i32, s.as_mut_ptr(), i.as_mut_ptr()) })?;
        Ok((0..q.n).map(|r| {
            let mut v: Vec<(f32, u32)> = (0..depth).filter(|&t| i[r * depth + t] != i32::MAX).map(|t| (s[r * depth + t], i[r * depth + t] as u32)).collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            v
        }).collect())
    }

    /// Mean squared distance of every row to its nearest centroid (exact f32).
    pub fn objective(&self, c: &[f32], k: usize) -> Result<f64> {
        Ok(self.assign(c, k)?.0)
    }

    /// (mean squared distance, nearest-centroid label per row), exact f32 SGEMM.
    pub fn assign(&self, c: &[f32], k: usize) -> Result<(f64, Vec<u32>)> {
        let mut labels = Vec::with_capacity(self.n);
        let mut tot = 0f64;
        const B: usize = 32_768; // bounds the m×k score scratch (512 MiB at k = 4096)
        for s0 in (0..self.n).step_by(B) {
            let m = (self.n - s0).min(B);
            let rows: Vec<i32> = (s0 as i32..(s0 + m) as i32).collect();
            let mut lab = vec![0i32; m];
            let mut dist = 0f64;
            check(unsafe { fc_batch_stats(self.dev, self.d as i32, rows.as_ptr(), m as i64, c.as_ptr(), k as i32, lab.as_mut_ptr(), &mut dist, std::ptr::null_mut(), std::ptr::null_mut()) })?;
            tot += dist;
            labels.extend(lab.into_iter().map(|v| v as u32));
        }
        Ok((tot / self.n as f64, labels))
    }
}

/// e-means on the device: centroids, batch sums and closing sums stay in GPU memory; per step only the batch row ids
/// go up and per-cluster counts / weights cross the bus (the algorithm itself is `kmeans::emeans_with`).
pub struct GpuEngine<'g, 'a> {
    g: &'g Gpu<'a>,
    k: usize,
    mode: i32, // assignment GEMM: 0 f32, 1 tf32, 2 f16 inputs, 3 bf16 inputs (f32 accumulate and updates)
}

impl<'g, 'a> GpuEngine<'g, 'a> {
    pub fn new(g: &'g Gpu<'a>) -> Self {
        GpuEngine { g, k: 0, mode: 0 }
    }
    pub fn with_precision(g: &'g Gpu<'a>, p: &str) -> Result<Self> {
        let mode = match p { "f32" => 0, "tf32" => 1, "f16" => 2, "bf16" => 3, "int8" => 4, "fp8" => 5, _ => bail!("unknown e-means precision {p}") };
        Ok(GpuEngine { g, k: 0, mode })
    }
}

impl EmeansEngine for GpuEngine<'_, '_> {
    fn n(&self) -> usize {
        self.g.n
    }
    fn d(&self) -> usize {
        self.g.d
    }
    fn rows(&self, idx: &[u32]) -> Vec<f32> {
        let d = self.g.d;
        idx.iter().flat_map(|&i| self.g.host[i as usize * d..(i as usize + 1) * d].iter().map(|v| v.to_f32())).collect()
    }
    fn set_centroids(&mut self, c: &[f32], k: usize) {
        self.k = k;
        check(unsafe { fc_em_init(c.as_ptr(), k as i32, self.g.d as i32, self.mode) }).expect("gpu e-means init");
    }
    fn batch(&mut self, batch: &[u32], closing: bool) -> (Vec<u32>, f64) {
        let rows: Vec<i32> = batch.iter().map(|&r| r as i32).collect();
        let (mut cnt, mut dist) = (vec![0u32; self.k], 0f64);
        check(unsafe { fc_em_batch(self.g.dev, rows.as_ptr(), rows.len() as i64, closing as i32, cnt.as_mut_ptr(), &mut dist) })
            .expect("gpu e-means batch");
        (cnt, dist)
    }
    fn update(&mut self, w: &[f32]) {
        check(unsafe { fc_em_update(w.as_ptr()) }).expect("gpu e-means update");
    }
    fn split(&mut self, pairs: &[(usize, usize)]) {
        let flat: Vec<i32> = pairs.iter().flat_map(|&(s, t)| [s as i32, t as i32]).collect();
        check(unsafe { fc_em_split(flat.as_ptr(), pairs.len() as i32) }).expect("gpu e-means split");
    }
    fn finalize(&mut self, pass_counts: &[f64]) -> Vec<f32> {
        let mut c = vec![0f32; self.k * self.g.d];
        check(unsafe { fc_em_finalize(pass_counts.as_ptr(), c.as_mut_ptr()) }).expect("gpu e-means finalize");
        c
    }
}

#[derive(clap::Args)]
pub struct GpuBenchArgs {
    /// dir written by `flashcluster prep`
    #[arg(long)]
    pub prep: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// corpus storage/compute precision for search: f16, bf16, f16acc, int8, fp8
    #[arg(long, default_value = "f16")]
    pub precision: String,
    /// re-rank a shortlist of this many candidates exactly on the device, reading f16 rows from pinned host memory (0 = off)
    #[arg(long, default_value_t = 0)]
    pub rerank: usize,
    /// re-rank from a device-resident f16 copy (speed mode) instead of pinned host memory (low-memory mode)
    #[arg(long)]
    pub rerank_device: bool,
    /// also fit e-means (quality law) on the GPU
    #[arg(long)]
    pub emeans: bool,
    /// e-means assignment GEMM precision: f32, tf32, f16, bf16, int8, fp8 (updates are always f32)
    #[arg(long, default_value = "f32")]
    pub emeans_precision: String,
    /// alias for --emeans-precision tf32
    #[arg(long)]
    pub tf32: bool,
    /// write the e-means run's final labels (u32 .npy) here
    #[arg(long)]
    pub labels_out: Option<PathBuf>,
    /// labels (.npy u32) of a reference run, for the ARI of this run's assignments
    #[arg(long)]
    pub ari_ref: Option<PathBuf>,
    #[arg(long, default_value = "1,64,256,2048")]
    pub batches: String,
    #[arg(long)]
    pub out: PathBuf,
}

pub fn bench(a: GpuBenchArgs) -> Result<()> {
    let name = init(a.device, false)?;
    let p = Prep::load(&a.prep)?;
    let (q, depth) = (&p.queries, p.depth);
    let prec: Precision = a.precision.parse()?;
    let fetch = depth;
    let t0 = Instant::now();
    let g = Gpu::upload_variant(&p.data, p.n, p.d, prec)?;
    let upload_s = t0.elapsed().as_secs_f64();
    if a.rerank > 0 {
        if a.rerank_device { g.rerank_device_copy()?; } else { g.register_host()?; }
    }
    let run = |qq: &Mat| -> Result<Vec<Vec<(f32, u32)>>> {
        if a.rerank > 0 { g.search_rerank(qq, a.rerank.max(depth), depth) } else { g.exact_search(qq, fetch) }
    };
    run(&q.gather(&[0]))?; // warm-up (cuBLAS heuristics, allocations)
    let mut rows = vec![];
    for b in a.batches.split(',').map(|s| s.parse::<usize>()) {
        let b = b?;
        // one untimed call at this batch size (cuBLAS kernel choice for the shape, buffer growth)
        run(&q.gather(&(0..b.min(q.n) as u32).collect::<Vec<_>>()))?;
        let t0 = Instant::now();
        let mut hits: Vec<Vec<u32>> = Vec::with_capacity(q.n);
        for s0 in (0..q.n).step_by(b) {
            let idx: Vec<u32> = (s0 as u32..(s0 + b).min(q.n) as u32).collect();
            hits.extend(run(&q.gather(&idx))?.into_iter().map(|h| h.into_iter().map(|x| x.1).collect::<Vec<_>>()));
        }
        let secs = t0.elapsed().as_secs_f64();
        let row = json!({"batch": b, "qps": q.n as f64 / secs, "ms_per_batch": secs * 1000.0 / q.n.div_ceil(b) as f64,
                         "metrics": metrics(&hits, &p)});
        println!("{row}");
        rows.push(row);
    }
    let mut report = json!({"engine": "flashcluster GPU exact search", "precision": a.precision, "rerank": a.rerank, "rerank_device": a.rerank_device, "device": name,
        "n": p.n, "d": p.d, "queries": q.n, "upload_seconds": upload_s, "device_mem_gb": device_mem_used_gb(),
        "peak_rss_gb": io::peak_rss_gb(), "search": rows});
    if a.emeans {
        let k = p.k;
        let ep = if a.tf32 { "tf32".to_string() } else { a.emeans_precision.clone() };
        let t0 = Instant::now();
        let (c, recipe) = emeans_with(&mut GpuEngine::with_precision(&g, &ep)?, k, 5.0, BatchLaw::Quality, 1234);
        let fit_s = t0.elapsed().as_secs_f64();
        let (obj, labels) = g.assign(&c, k)?;
        let mut e = json!({"k": k, "emeans_precision": ep, "gpu_fit_seconds": fit_s, "objective": obj,
            "objective_prep_centroids": g.objective(&p.centroids, k)?, "recipe": recipe});
        if let Some(path) = &a.labels_out {
            io::write_npy_bytes(path, "<u4", &[labels.len()], io::as_bytes(&labels))?;
        }
        if let Some(path) = &a.ari_ref {
            let (_, r) = io::read_npy_u32(path)?;
            e["ari_vs_ref"] = json!(crate::kmeans::adjusted_rand(&labels, &r));
        }
        println!("{e}");
        report["emeans"] = e;
    }
    report["device_mem_gb"] = json!(device_mem_used_gb());
    io::write_json(&a.out, &report)?;
    Ok(())
}
