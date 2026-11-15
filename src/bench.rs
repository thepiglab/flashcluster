//! Benchmark inputs (`prep`) and the CPU search benchmark (`search-bench`).
//!
//! A prep directory is what every engine (flashcluster CPU, flashcluster GPU, and any reference harness) reads, so
//! partitions, data and ground truth are identical across engines:
//!   data.f16.npy        n × d corpus (f16; rows L2-normalised unless --no-normalize)
//!   queries.f32.npy     nq × d queries (f32, normalised the same way)
//!   centroids.f32.npy   k × d shared IVF centroids (e-means, quality law)
//!   exact.u32.npy       nq × depth exact inner-product top-depth row ids (ground truth)
//!   groups.u32.npy      optional: group id per row (e.g. document of each chunk) for group-level recall
//!   targets.u32.npy     optional: target per query (a group id if groups exist, else a row id)
//!   meta.json           {"depth", "grouped", ...}
//! Older prep dirs (pool.f16.npy, pool_article.u32.npy, "article_level") are read too.
use crate::index::Ivf;
use crate::io::{self, Mat};
use crate::kmeans::{self, Backend};
use crate::linalg::flat_search;
use anyhow::{bail, Result};
use half::f16;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Prep {
    pub data: Vec<f16>,
    pub n: usize,
    pub d: usize,
    pub queries: Mat,
    pub centroids: Vec<f32>,
    pub k: usize,
    pub exact: Vec<u32>,
    pub depth: usize,
    pub groups: Option<Vec<u32>>,
    pub targets: Option<Vec<u32>>,
    pub load_seconds: f64,
}

fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

impl Prep {
    pub fn load(dir: &Path) -> Result<Prep> {
        let meta: Value = io::read_json(&dir.join("meta.json"))?;
        let depth = meta["depth"].as_u64().unwrap_or(100) as usize;
        let grouped = meta["grouped"].as_bool().or(meta["article_level"].as_bool()).unwrap_or(false);
        let (cs, centroids) = io::read_npy_f32(&dir.join("centroids.f32.npy"))?;
        let (qs, qv) = io::read_npy_f32(&dir.join("queries.f32.npy"))?;
        let (_, exact) = io::read_npy_u32(&dir.join("exact.u32.npy"))?;
        let groups = match first_existing(dir, &["groups.u32.npy", "pool_article.u32.npy"]) {
            Some(p) if grouped => Some(io::read_npy_u32(&p)?.1),
            _ => None,
        };
        let targets = match first_existing(dir, &["targets.u32.npy"]) {
            Some(p) => Some(io::read_npy_u32(&p)?.1),
            None => None,
        };
        let t0 = Instant::now();
        let Some(dp) = first_existing(dir, &["data.f16.npy", "pool.f16.npy"]) else { bail!("no data.f16.npy in {}", dir.display()) };
        let (ds, data) = io::read_npy_f16(&dp)?;
        let load_seconds = t0.elapsed().as_secs_f64();
        if ds[1] != qs[1] || cs[1] != qs[1] {
            bail!("dimension mismatch");
        }
        Ok(Prep { data, n: ds[0], d: ds[1], queries: Mat { data: qv, n: qs[0], d: qs[1] }, centroids, k: cs[0], exact, depth,
                  groups, targets, load_seconds })
    }
}

/// overlap@10 with the exact top-10, and (if targets exist) hit rates R@1/10/100: group-level (rank among distinct
/// groups) when groups exist, else row-level.
pub fn metrics(hits: &[Vec<u32>], p: &Prep) -> Value {
    let n = hits.len() as f64;
    let ov: f64 = hits.iter().enumerate().map(|(i, h)| {
        let e = &p.exact[i * p.depth..i * p.depth + 10];
        h.iter().take(10).filter(|x| e.contains(x)).count() as f64 / 10.0
    }).sum::<f64>() / n;
    let ov100: f64 = hits.iter().enumerate().map(|(i, h)| {
        let e = &p.exact[i * p.depth..i * p.depth + p.depth.min(100)];
        h.iter().take(100).filter(|x| e.contains(x)).count() as f64 / p.depth.min(100) as f64
    }).sum::<f64>() / n;
    let mut out = json!({"overlap@10": ov, "recall@100": ov100});
    if let Some(t) = &p.targets {
        let mut ranks = vec![];
        for (row, &tg) in hits.iter().zip(t) {
            let hit = match &p.groups {
                Some(g) => {
                    let mut seen: Vec<u32> = vec![];
                    let mut hit = None;
                    for &j in row {
                        let a = g[j as usize];
                        if !seen.contains(&a) {
                            seen.push(a);
                            if a == tg {
                                hit = Some(seen.len());
                                break;
                            }
                        }
                    }
                    hit
                }
                None => row.iter().position(|&j| j == tg).map(|x| x + 1),
            };
            ranks.push(hit.map_or(f64::INFINITY, |h| h as f64));
        }
        let at = |k: f64| ranks.iter().filter(|&&r| r <= k).count() as f64 / n;
        out["R@1"] = json!(at(1.0));
        out["R@10"] = json!(at(10.0));
        out["R@100"] = json!(at(100.0));
    }
    out
}

#[derive(clap::Args)]
pub struct PrepArgs {
    /// corpus .npy (n × d, f16/f32/f64)
    #[arg(long)]
    pub data: PathBuf,
    /// queries .npy (nq × d)
    #[arg(long)]
    pub queries: PathBuf,
    /// optional group id per corpus row (.npy u4)
    #[arg(long)]
    pub groups: Option<PathBuf>,
    /// optional target per query (.npy u4): group id with --groups, else row id
    #[arg(long)]
    pub targets: Option<PathBuf>,
    #[arg(long, default_value_t = 4096)]
    pub k: usize,
    #[arg(long, default_value_t = 100)]
    pub depth: usize,
    /// keep vectors as given (default: L2-normalise rows, i.e. cosine)
    #[arg(long)]
    pub no_normalize: bool,
    #[arg(long)]
    pub out: PathBuf,
}

pub fn prep(a: PrepArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let (ds, dv) = io::read_npy_f32(&a.data)?;
    let mut x = Mat { data: dv, n: ds[0], d: ds[1] };
    let (qs, qv) = io::read_npy_f32(&a.queries)?;
    let mut q = Mat { data: qv, n: qs[0], d: qs[1] };
    if !a.no_normalize {
        x.normalise();
        q.normalise();
    }
    let x16: Vec<f16> = x.data.par_iter().map(|&v| f16::from_f32(v)).collect();
    x.data.par_iter_mut().zip(x16.par_iter()).for_each(|(a, b)| *a = b.to_f32()); // every engine sees the f16 values
    io::write_npy_bytes(&a.out.join("data.f16.npy"), "<f2", &[x.n, x.d], io::as_bytes(&x16))?;
    drop(x16);
    io::write_npy_bytes(&a.out.join("queries.f32.npy"), "<f4", &[q.n, q.d], io::as_bytes(&q.data))?;
    let t0 = Instant::now();
    let c = kmeans::fit(&x, a.k, &Backend::emeans_quality(), 1234)?;
    let fit_s = t0.elapsed().as_secs_f64();
    io::write_npy_bytes(&a.out.join("centroids.f32.npy"), "<f4", &[a.k, x.d], io::as_bytes(&c.centroids))?;
    let ex: Vec<u32> = flat_search(&x, &q, a.depth).iter().flat_map(|r| r.iter().map(|t| t.1)).collect();
    io::write_npy_bytes(&a.out.join("exact.u32.npy"), "<u4", &[q.n, a.depth], io::as_bytes(&ex))?;
    for (src, name) in [(&a.groups, "groups.u32.npy"), (&a.targets, "targets.u32.npy")] {
        if let Some(p) = src {
            std::fs::copy(p, a.out.join(name))?;
        }
    }
    let meta = json!({"n": x.n, "d": x.d, "k": a.k, "queries": q.n, "depth": a.depth, "grouped": a.groups.is_some(),
                      "normalized": !a.no_normalize, "centroids": "e-means quality law, seed 1234", "fit_seconds": fit_s});
    io::write_json(&a.out.join("meta.json"), &meta)?;
    println!("{meta}");
    Ok(())
}

#[derive(clap::Args)]
pub struct SearchBenchArgs {
    #[arg(long)]
    pub prep: PathBuf,
    #[arg(long, default_value = "8,16,32,64,128")]
    pub nprobe: String,
    /// timed repetitions of the batch run (median reported)
    #[arg(long, default_value_t = 3)]
    pub reps: usize,
    /// store the lists as int8 with a per-vector scale (half the bytes of f16; approximate scores)
    #[arg(long)]
    pub int8: bool,
    /// re-rank a shortlist of this many exactly from the memory-mapped f16 data file (0 = off)
    #[arg(long, default_value_t = 0)]
    pub rerank: usize,
    #[arg(long)]
    pub out: PathBuf,
}

/// Memory-mapped f16 corpus (the prep data file), for exact re-ranking without a second in-memory copy.
pub struct MappedF16 {
    map: memmap2::Mmap,
    off: usize,
    d: usize,
}

impl MappedF16 {
    pub fn open(dir: &Path, d: usize) -> Result<Self> {
        let path = first_existing(dir, &["data.f16.npy", "pool.f16.npy"]).ok_or_else(|| anyhow::anyhow!("no data file"))?;
        let f = std::fs::File::open(&path)?;
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let hlen = u16::from_le_bytes([map[8], map[9]]) as usize;
        Ok(MappedF16 { map, off: 10 + hlen, d })
    }
    pub fn row(&self, i: usize) -> &[f16] {
        let b = &self.map[self.off + i * self.d * 2..self.off + (i + 1) * self.d * 2];
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f16, self.d) }
    }
}

/// Exact re-rank of shortlists; best `depth` ids per query.
pub fn rerank_rows(m: &MappedF16, q: &Mat, qidx: &[usize], short: &[Vec<(f32, u32)>], depth: usize) -> Vec<Vec<u32>> {
    short.par_iter().zip(qidx.par_iter()).map(|(cands, &qi)| {
        let qr = q.row(qi);
        let mut s: Vec<(f32, u32)> = cands.iter().map(|&(_, id)| {
            let mut sc = [0f32; crate::simd::R];
            crate::simd::dot_rows_f16(qr, m.row(id as usize), 1, &mut sc);
            (sc[0], id)
        }).collect();
        s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        s.into_iter().take(depth).map(|x| x.1).collect()
    }).collect()
}

/// CPU IVF: per nprobe, single-thread latency (one query per call), single-thread batch q/s, all-thread batch q/s.
pub fn search_bench(a: SearchBenchArgs) -> Result<()> {
    let mut p = Prep::load(&a.prep)?;
    let t0 = Instant::now();
    let data = std::mem::take(&mut p.data); // moved into the index (built in place): one copy of the corpus
    let mut ivf = Ivf::build_f16(data, p.d, &p.centroids, p.k);
    if a.int8 {
        ivf.quantize_int8();
    }
    let build_s = t0.elapsed().as_secs_f64();
    let q = &p.queries;
    let mapped = if a.rerank > 0 { Some(MappedF16::open(&a.prep, p.d)?) } else { None };
    let fetch = if a.rerank > 0 { a.rerank.max(p.depth) } else { p.depth };
    let finish = |res: Vec<Vec<(f32, u32)>>, qidx: &[usize]| -> Vec<Vec<u32>> {
        match &mapped {
            Some(m) => rerank_rows(m, q, qidx, &res, p.depth),
            None => res.into_iter().map(|h| h.into_iter().map(|x| x.1).collect()).collect(),
        }
    };
    let all: Vec<usize> = (0..q.n).collect();
    let threads = rayon::current_num_threads();
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let mut rows = vec![];
    for np in a.nprobe.split(',').map(|s| s.parse::<usize>()) {
        let np = np?;
        let t0 = Instant::now();
        let lat: Vec<Vec<u32>> = one.install(|| (0..q.n).map(|i| finish(vec![ivf.search_one(q.row(i), np, fetch)], &[i]).remove(0)).collect());
        let lat_ms = t0.elapsed().as_secs_f64() * 1000.0 / q.n as f64;
        let t0 = Instant::now();
        drop(one.install(|| finish(ivf.search(q, np, fetch), &all)));
        let b1_qps = q.n as f64 / t0.elapsed().as_secs_f64();
        let (mut times, mut ids) = (vec![], vec![]);
        for _ in 0..a.reps.max(1) {
            let t0 = Instant::now();
            ids = finish(ivf.search(q, np, fetch), &all);
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let same = ids.iter().zip(&lat).filter(|(a, b)| a.iter().take(10).eq(b.iter().take(10))).count() as f64 / q.n as f64;
        let row = json!({"nprobe": np, "metrics": metrics(&ids, &p), "latency_ms_1thread": lat_ms, "batch_qps_1thread": b1_qps,
            "batch_qps": q.n as f64 / times[times.len() / 2], "batch_threads": threads, "batch_vs_latency_top10_identical": same});
        println!("{row}");
        rows.push(row);
    }
    let report = json!({"engine": "flashcluster CPU IVF", "storage": if a.int8 { "int8" } else { "f16" }, "rerank": a.rerank,
        "simd": format!("{:?}", crate::simd::level()), "int8_simd": format!("{:?}", crate::quant::level()),
        "n": p.n, "d": p.d, "k": p.k, "queries": q.n, "load_seconds": p.load_seconds, "build_seconds": build_s,
        "index_bytes": ivf.memory_bytes(), "peak_rss_gb": io::peak_rss_gb(), "sweep": rows});
    io::write_json(&a.out, &report)?;
    println!("{}", json!({"build_seconds": build_s, "index_gb": ivf.memory_bytes() as f64 / 1e9, "peak_rss_gb": report["peak_rss_gb"], "simd": report["simd"], "int8_simd": report["int8_simd"]}));
    Ok(())
}

#[derive(clap::Args)]
pub struct FitArgs {
    /// corpus .npy (n × d)
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub k: usize,
    /// emeans (default) or lloyd
    #[arg(long, default_value = "emeans")]
    pub backend: String,
    #[arg(long, default_value_t = 5.0)]
    pub nepochs: f64,
    /// e-means batch law: quality (default), throughput or auto
    #[arg(long, default_value = "quality")]
    pub batch_law: String,
    #[arg(long, default_value_t = 25)]
    pub niter: usize,
    #[arg(long, default_value_t = 1234)]
    pub seed: u64,
    #[arg(long)]
    pub no_normalize: bool,
    /// write centroids.f32.npy, labels.u32.npy and fit.json here
    #[arg(long)]
    pub out: PathBuf,
}

pub fn fit(a: FitArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let (ds, dv) = io::read_npy_f32(&a.data)?;
    let mut x = Mat { data: dv, n: ds[0], d: ds[1] };
    if !a.no_normalize {
        x.normalise();
    }
    let backend = match a.backend.as_str() {
        "emeans" => Backend::Emeans { nepochs: a.nepochs, batch_law: a.batch_law.parse()? },
        "lloyd" => Backend::Lloyd { niter: a.niter, max_points_per_centroid: Some(256) },
        b => bail!("unknown backend {b}"),
    };
    let c = kmeans::fit(&x, a.k, &backend, a.seed)?;
    io::write_npy_bytes(&a.out.join("centroids.f32.npy"), "<f4", &[a.k, x.d], io::as_bytes(&c.centroids))?;
    io::write_npy_bytes(&a.out.join("labels.u32.npy"), "<u4", &[x.n], io::as_bytes(&c.assign))?;
    let rep = json!({"seconds": c.seconds, "objective": kmeans::objective(&x, &c.centroids, a.k), "balance": kmeans::balance(&c.assign, a.k),
                     "params": c.params, "peak_rss_gb": io::peak_rss_gb()});
    io::write_json(&a.out.join("fit.json"), &rep)?;
    println!("{rep}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn prep_then_search_bench_end_to_end() {
        let dir = std::env::temp_dir().join(format!("flashcluster-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (n, d, nq) = (3000, 32, 50);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        let x: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>() - 0.5).collect();
        io::write_npy_bytes(&dir.join("x.npy"), "<f4", &[n, d], io::as_bytes(&x)).unwrap();
        // queries = the first rows (each finds itself); groups = row / 10, targets = their groups
        io::write_npy_bytes(&dir.join("q.npy"), "<f4", &[nq, d], io::as_bytes(&x[..nq * d])).unwrap();
        let g: Vec<u32> = (0..n as u32).map(|i| i / 10).collect();
        let t: Vec<u32> = (0..nq as u32).map(|i| i / 10).collect();
        io::write_npy_bytes(&dir.join("g.npy"), "<u4", &[n], io::as_bytes(&g)).unwrap();
        io::write_npy_bytes(&dir.join("t.npy"), "<u4", &[nq], io::as_bytes(&t)).unwrap();
        prep(PrepArgs { data: dir.join("x.npy"), queries: dir.join("q.npy"), groups: Some(dir.join("g.npy")), targets: Some(dir.join("t.npy")),
                        k: 16, depth: 100, no_normalize: false, out: dir.join("prep") }).unwrap();
        search_bench(SearchBenchArgs { prep: dir.join("prep"), nprobe: "16".into(), reps: 1, int8: false, rerank: 0, out: dir.join("r.json") }).unwrap();
        search_bench(SearchBenchArgs { prep: dir.join("prep"), nprobe: "16".into(), reps: 1, int8: true, rerank: 200, out: dir.join("r8.json") }).unwrap();
        let r8: Value = io::read_json(&dir.join("r8.json")).unwrap();
        assert_eq!(r8["sweep"][0]["metrics"]["overlap@10"], 1.0, "int8 + exact re-rank should recover the exact top-10");
        let r: Value = io::read_json(&dir.join("r.json")).unwrap();
        let m = &r["sweep"][0]["metrics"];
        assert_eq!(m["overlap@10"], 1.0, "{m}");
        assert_eq!(m["R@1"], 1.0, "{m}");
        assert_eq!(r["sweep"][0]["batch_vs_latency_top10_identical"], 1.0);
        fit(FitArgs { data: dir.join("x.npy"), k: 8, backend: "emeans".into(), nepochs: 5.0, batch_law: "quality".into(), niter: 25,
                      seed: 1, no_normalize: false, out: dir.join("fit") }).unwrap();
        assert!(dir.join("fit/labels.u32.npy").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
