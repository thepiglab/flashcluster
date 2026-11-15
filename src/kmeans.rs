//! k-means backends on identical inputs, plus quality measures recomputed the same way for every backend.
//!
//! * `Lloyd`: FAISS-style Lloyd k-means (random init from the data, empty clusters re-seeded by splitting a large
//!   cluster with a ±1/1024 perturbation, optional `max_points_per_centroid` subsample). The comparison arm.
//! * `Emeans`: e-means, Algorithm 1 of "E-Means: Scaling Laws for Mini-Batch K-Means" (arXiv 2611.09501), ported
//!   from the FAISS reference (`faiss/contrib/e_means`, faiss 1.16.0): EMA mini-batch updates with the fitted batch
//!   and step laws, cosine-annealed step to 0.1·α0, revival of under-used clusters every 5 updates by splitting
//!   mass-sampled donors, closing means from the last epoch's moving assignments. Same constants and laws; random
//!   streams differ (ChaCha8 instead of NumPy's MT19937), so runs are statistically, not bitwise, equivalent.
use crate::io::Mat;
use crate::linalg::{nearest, Metric};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BatchLaw {
    Auto,
    Throughput,
    Quality,
}

impl std::str::FromStr for BatchLaw {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "auto" => BatchLaw::Auto,
            "throughput" => BatchLaw::Throughput,
            "quality" => BatchLaw::Quality,
            _ => anyhow::bail!("unknown batch law {s}"),
        })
    }
}

#[derive(Clone, Debug)]
pub enum Backend {
    Lloyd { niter: usize, max_points_per_centroid: Option<usize> },
    Emeans { nepochs: f64, batch_law: BatchLaw },
}

impl Backend {
    pub fn emeans_quality() -> Self {
        Backend::Emeans { nepochs: 5.0, batch_law: BatchLaw::Quality }
    }
    pub fn name(&self) -> String {
        match self {
            Backend::Lloyd { .. } => "lloyd".into(),
            Backend::Emeans { .. } => "emeans".into(),
        }
    }
}

pub struct Clustering {
    pub centroids: Vec<f32>, // k×d
    pub k: usize,
    pub assign: Vec<u32>,    // exact nearest centroid (squared L2) for every row
    pub seconds: f64,        // training time (excludes the final full assignment)
    pub params: serde_json::Value,
}

pub fn fit(x: &Mat, k: usize, backend: &Backend, seed: u64) -> anyhow::Result<Clustering> {
    anyhow::ensure!(k >= 1 && k <= x.n, "need 1 <= k <= n (k={k}, n={})", x.n);
    let t0 = Instant::now();
    let (c, params) = match backend {
        Backend::Lloyd { niter, max_points_per_centroid } => {
            (lloyd(x, k, *niter, *max_points_per_centroid, seed), json!({"backend": "lloyd", "niter": niter,
                "max_points_per_centroid": max_points_per_centroid}))
        }
        Backend::Emeans { nepochs, batch_law } => {
            let (c, recipe) = emeans(x, k, *nepochs, *batch_law, seed);
            (c, json!({"backend": "emeans", "nepochs": nepochs, "batch_law": format!("{batch_law:?}").to_lowercase(),
                       "recipe": recipe}))
        }
    };
    let seconds = t0.elapsed().as_secs_f64();
    let (assign, _) = nearest(x, None, &c, k, Metric::L2);
    Ok(Clustering { centroids: c, k, assign, seconds,
        params: json!({"k": k, "seed": seed, "n": x.n, "d": x.d, "backend": params}) })
}

/// Rows grouped by label: (order, starts) with rows of cluster j at order[starts[j]..starts[j+1]].
pub fn group_by_label(labels: &[u32], k: usize) -> (Vec<u32>, Vec<usize>) {
    let mut starts = vec![0usize; k + 1];
    for &l in labels {
        starts[l as usize + 1] += 1;
    }
    for j in 0..k {
        starts[j + 1] += starts[j];
    }
    let mut fill = starts.clone();
    let mut order = vec![0u32; labels.len()];
    for (i, &l) in labels.iter().enumerate() {
        order[fill[l as usize]] = i as u32;
        fill[l as usize] += 1;
    }
    (order, starts)
}

/// Per-cluster sums (k×d) and counts of rows `rows[t]` (or t) carrying label `labels[t]`.
fn sums_counts(x: &Mat, rows: Option<&[u32]>, labels: &[u32], k: usize) -> (Vec<f32>, Vec<u32>) {
    let d = x.d;
    let (order, starts) = group_by_label(labels, k);
    let mut sums = vec![0f32; k * d];
    sums.par_chunks_mut(d).enumerate().for_each(|(j, s)| {
        for &t in &order[starts[j]..starts[j + 1]] {
            let i = rows.map_or(t as usize, |r| r[t as usize] as usize);
            for (a, b) in s.iter_mut().zip(x.row(i)) {
                *a += b;
            }
        }
    });
    let counts = (0..k).map(|j| (starts[j + 1] - starts[j]) as u32).collect();
    (sums, counts)
}

// ---------------------------------------------------------------- Lloyd (FAISS-style)

pub fn lloyd(x: &Mat, k: usize, niter: usize, max_ppc: Option<usize>, seed: u64) -> Vec<f32> {
    let d = x.d;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let train_owned;
    let xt: &Mat = match max_ppc {
        Some(m) if x.n > m * k => {
            let mut idx = rand::seq::index::sample(&mut rng, x.n, m * k).into_vec();
            idx.sort_unstable();
            let idx: Vec<u32> = idx.into_iter().map(|i| i as u32).collect();
            train_owned = x.gather(&idx);
            &train_owned
        }
        _ => x,
    };
    let init: Vec<u32> = rand::seq::index::sample(&mut rng, xt.n, k).into_iter().map(|i| i as u32).collect();
    let mut c = xt.gather(&init).data;
    const EPS: f32 = 1.0 / 1024.0;
    for _ in 0..niter {
        let (lab, _) = nearest(xt, None, &c, k, Metric::L2);
        let (sums, counts) = sums_counts(xt, None, &lab, k);
        let mut hassign: Vec<f64> = counts.iter().map(|&v| v as f64).collect();
        for j in 0..k {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f32;
                for t in 0..d {
                    c[j * d + t] = sums[j * d + t] * inv;
                }
            }
        }
        // FAISS split_clusters: re-seed each empty cluster from a cluster drawn in proportion to (size - 1)
        let n = xt.n as f64;
        for ci in 0..k {
            if hassign[ci] != 0.0 {
                continue;
            }
            let mut cj = 0usize;
            loop {
                let p = (hassign[cj] - 1.0) / (n - k as f64);
                if rng.gen::<f64>() < p {
                    break;
                }
                cj = (cj + 1) % k;
            }
            for t in 0..d {
                let v = c[cj * d + t];
                let (a, b) = if t % 2 == 0 { (1.0 + EPS, 1.0 - EPS) } else { (1.0 - EPS, 1.0 + EPS) };
                c[ci * d + t] = v * a;
                c[cj * d + t] = v * b;
            }
            hassign[ci] = hassign[cj] / 2.0;
            hassign[cj] -= hassign[ci];
        }
    }
    c
}

// ---------------------------------------------------------------- e-means (port of faiss.contrib.e_means)

const ALPHA_MIN_FACTOR: f64 = 0.1;
const CHECK_UNUSED_EVERY: usize = 5;
const INIT_USAGE: f32 = 1.0 / 65536.0;
const SPLIT_EPSILON: f32 = 1e-4;
const UNUSED_THRESHOLD: f32 = 0.1;
const NO_REVIVAL_MAX_D: usize = 2;
const AUTO_QUALITY_MAX_D: usize = 16;
const BATCH_THROUGHPUT: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const BATCH_QUALITY: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

fn round_pow2(v: f64) -> usize {
    if v <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(v.log2().floor() as i32);
    let high = 2.0 * low;
    (if v / low < high / v { low } else { high }) as usize
}

pub fn step_law(n: usize, k: usize, nepochs: f64, b: usize) -> f64 {
    let (c, ek, eb, en, et) = ALPHA_LAW;
    (c * (k as f64).powf(ek) * (b as f64).powf(eb) * (n as f64).powf(en) * nepochs.powf(et)).clamp(1e-4, 1.0)
}

pub fn resolve_law(law: BatchLaw, d: usize) -> BatchLaw {
    match law {
        BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
        BatchLaw::Auto => BatchLaw::Throughput,
        l => l,
    }
}

/// (batch size B, alpha0) from the fitted batch and step laws.
pub fn recipe(n: usize, k: usize, nepochs: f64, law: BatchLaw) -> (usize, f64) {
    let (c, ek, en, et) = if law == BatchLaw::Quality { BATCH_QUALITY } else { BATCH_THROUGHPUT };
    let raw = c * (k as f64).powf(ek) * (n as f64).powf(en) * nepochs.powf(et);
    let mut b = round_pow2(raw);
    b = if law == BatchLaw::Throughput { b.min(round_pow2(n as f64 / 2.0)) } else { b.min(n / 2) };
    b = b.min(n).max(1);
    (b, step_law(n, k, nepochs, b))
}

pub fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor() as usize;
    let rem = ((nepochs - whole as f64) * n as f64).round_ties_even() as usize;
    if whole == 0 {
        return vec![rem];
    }
    let mut v = vec![n; whole - 1];
    if rem > 0 {
        v.push(rem);
    }
    v.push(n);
    v
}

fn scheduled_alpha(alpha0: f64, step: usize, total: usize) -> f64 {
    let f = step.min(total) as f64 / total.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * f).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Returns (centroids, recipe JSON).
/// Where the e-means arithmetic runs. The algorithm (schedule, masses, revival choices, closing) lives in
/// `emeans_with` and is shared; an engine only holds the centroids and does the per-batch work:
/// assign a batch and keep its per-cluster sums (`batch`), move centroids toward the batch means (`update`), apply
/// donor splits (`split`), and produce the closing means (`finalize`). CPU: `CpuEngine`; GPU: `gpu::GpuEngine`,
/// which keeps centroids and sums on the device so only per-cluster counts and weights cross the bus.
pub trait EmeansEngine {
    fn n(&self) -> usize;
    fn d(&self) -> usize;
    /// Rows `idx` as an f32 matrix (initial centroids).
    fn rows(&self, idx: &[u32]) -> Vec<f32>;
    fn set_centroids(&mut self, c: &[f32], k: usize);
    /// Assign `batch` to the current centroids; keep per-cluster sums; when `closing`, also add them to the closing
    /// sums. Returns (per-cluster counts, sum of squared distances).
    fn batch(&mut self, batch: &[u32], closing: bool) -> (Vec<u32>, f64);
    /// c_j += w_j (mean_j - c_j) for clusters with a non-zero count in the last batch (w_j == 1: c_j = mean_j).
    fn update(&mut self, w: &[f32]);
    /// Donor splits (Appendix J): for each (src, dst), c_src *= 1 - off, c_dst = c_src_old * (1 + off), off = ±1e-4
    /// alternating by coordinate; identical results nudged apart in coordinate 0.
    fn split(&mut self, pairs: &[(usize, usize)]);
    /// Closing means: c_j = closing_sum_j / pass_counts_j where pass_counts_j > 0; returns all centroids.
    fn finalize(&mut self, pass_counts: &[f64]) -> Vec<f32>;
}

pub struct CpuEngine<'a> {
    x: &'a Mat,
    c: Vec<f32>,
    k: usize,
    sums: Vec<f32>,
    counts: Vec<u32>,
    closing: Vec<f32>,
}

impl<'a> CpuEngine<'a> {
    pub fn new(x: &'a Mat) -> Self {
        CpuEngine { x, c: vec![], k: 0, sums: vec![], counts: vec![], closing: vec![] }
    }
}

/// Apply one donor split in place (shared by the CPU engine and tests).
pub fn split_rows(c: &mut [f32], d: usize, src: usize, dst: usize) {
    let mut collapsed = true;
    for i in 0..d {
        let off = if i % 2 == 0 { SPLIT_EPSILON } else { -SPLIT_EPSILON };
        let v = c[src * d + i];
        let (a, b) = (v * (1.0 - off), v * (1.0 + off));
        collapsed &= a == b;
        c[src * d + i] = a;
        c[dst * d + i] = b;
    }
    if collapsed {
        let v = c[dst * d];
        let up = next_up(v);
        c[dst * d] = if up.is_finite() { up } else { -next_up(-v) };
    }
}

impl EmeansEngine for CpuEngine<'_> {
    fn n(&self) -> usize {
        self.x.n
    }
    fn d(&self) -> usize {
        self.x.d
    }
    fn rows(&self, idx: &[u32]) -> Vec<f32> {
        self.x.gather(idx).data
    }
    fn set_centroids(&mut self, c: &[f32], k: usize) {
        self.c = c.to_vec();
        self.k = k;
        self.closing = vec![0f32; k * self.x.d];
    }
    fn batch(&mut self, batch: &[u32], closing: bool) -> (Vec<u32>, f64) {
        let (lab, dist) = nearest(self.x, Some(batch), &self.c, self.k, Metric::L2);
        let (sums, counts) = sums_counts(self.x, Some(batch), &lab, self.k);
        if closing {
            self.closing.par_iter_mut().zip(sums.par_iter()).for_each(|(a, b)| *a += b);
        }
        self.sums = sums;
        self.counts = counts.clone();
        (counts, dist.iter().map(|&v| v as f64).sum())
    }
    fn update(&mut self, w: &[f32]) {
        let (d, sums, counts) = (self.x.d, &self.sums, &self.counts);
        self.c.par_chunks_mut(d).enumerate().for_each(|(j, cj)| {
            let nj = counts[j];
            if nj == 0 {
                return;
            }
            let inv = 1.0 / nj as f32;
            let s = &sums[j * d..(j + 1) * d];
            for t in 0..d {
                let mean = s[t] * inv;
                cj[t] = if w[j] == 1.0 { mean } else { cj[t] + w[j] * (mean - cj[t]) };
            }
        });
    }
    fn split(&mut self, pairs: &[(usize, usize)]) {
        for &(s, t) in pairs {
            split_rows(&mut self.c, self.x.d, s, t);
        }
    }
    fn finalize(&mut self, pass_counts: &[f64]) -> Vec<f32> {
        let d = self.x.d;
        for j in 0..self.k {
            if pass_counts[j] > 0.0 {
                let inv = (1.0 / pass_counts[j]) as f32;
                for t in 0..d {
                    self.c[j * d + t] = self.closing[j * d + t] * inv;
                }
            }
        }
        std::mem::take(&mut self.c)
    }
}

pub fn emeans(x: &Mat, k: usize, nepochs: f64, law: BatchLaw, seed: u64) -> (Vec<f32>, serde_json::Value) {
    emeans_with(&mut CpuEngine::new(x), k, nepochs, law, seed)
}

/// e-means on any engine: the update rule (Equation 2 in mass/position form), cosine schedule, revival and closing
/// are decided here, so CPU and GPU runs follow one implementation.
pub fn emeans_with<E: EmeansEngine>(eng: &mut E, k: usize, nepochs: f64, law: BatchLaw, seed: u64) -> (Vec<f32>, serde_json::Value) {
    let (n, d) = (eng.n(), eng.d());
    let law = resolve_law(law, d);
    let (bs, alpha0) = recipe(n, k, nepochs, law);
    let rows = epoch_rows(n, nepochs);
    let steps: Vec<usize> = rows.iter().map(|r| r.div_ceil(bs)).collect();
    let total: usize = steps.iter().sum();
    let tau = if d <= NO_REVIVAL_MAX_D { 0.0 } else { UNUSED_THRESHOLD };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let init: Vec<u32> = if n == k { (0..n as u32).collect() } else {
        rand::seq::index::sample(&mut rng, n, k).into_iter().map(|i| i as u32).collect()
    };
    eng.set_centroids(&eng.rows(&init), k);
    let mut usage = vec![INIT_USAGE; k];
    let mut split_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    let (mut step, mut nsplits) = (0usize, 0usize);
    let mut objectives = vec![];
    let last = rows.len() - 1;
    let mut pass_counts = vec![0f64; k];
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut w = vec![0f32; k];
    for (e, &erows) in rows.iter().enumerate() {
        pass_counts.iter_mut().for_each(|v| *v = 0.0);
        let mut obj = 0f64;
        let es = ((seed.wrapping_mul(1000003)).wrapping_add(e as u64)) % (1u64 << 32);
        perm.iter_mut().enumerate().for_each(|(i, v)| *v = i as u32);
        perm.shuffle(&mut ChaCha8Rng::seed_from_u64(es));
        for b0 in (0..erows).step_by(bs) {
            let batch = &perm[b0..(b0 + bs).min(erows)];
            let (counts, dist_sum) = eng.batch(batch, e == last);
            obj += dist_sum;
            // Algorithm 1 line 8 (Equation 2), in mass/position form
            let alpha = scheduled_alpha(alpha0, step, total);
            let replace = alpha == 1.0;
            if !replace {
                let decay = (1.0 - alpha) as f32;
                usage.iter_mut().for_each(|u| *u *= decay);
            }
            let alpha32 = alpha as f32;
            for j in 0..k {
                let nj = counts[j];
                pass_counts[j] += nj as f64;
                if nj == 0 {
                    w[j] = 0.0;
                    continue;
                }
                let retained = if replace { 0.0 } else { usage[j] };
                let mass = retained + alpha32 * nj as f32;
                w[j] = alpha32 * nj as f32 / mass;
                usage[j] = mass;
            }
            eng.update(&w);
            if (step + 1) % CHECK_UNUSED_EVERY == 0 && tau > 0.0 {
                let pairs = choose_splits(&mut usage, tau, &mut split_rng);
                nsplits += pairs.len();
                if !pairs.is_empty() {
                    eng.split(&pairs);
                }
            }
            step += 1;
        }
        objectives.push(obj);
    }
    let c = eng.finalize(&pass_counts);
    (c, json!({"batch_law": format!("{law:?}").to_lowercase(), "batch_size": bs, "alpha0": alpha0,
               "epoch_rows": rows, "total_steps": total, "splits": nsplits, "moving_objective": objectives}))
}

/// Algorithm 1 line 9 / Appendix J: clusters below tau·mean(mass) take half of a donor's mass, donors sampled
/// without replacement in proportion to mass (Gumbel top-k). Updates the masses; returns (donor, target) pairs.
fn choose_splits(usage: &mut [f32], tau: f32, rng: &mut ChaCha8Rng) -> Vec<(usize, usize)> {
    let k = usage.len();
    let thr = usage.iter().sum::<f32>() * tau / k as f32;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < thr).collect();
    let donors: Vec<usize> = (0..k).filter(|&j| usage[j] >= thr && usage[j] > 0.0).collect();
    let nsplit = unused.len().min(donors.len());
    if nsplit == 0 {
        return vec![];
    }
    let mut keyed: Vec<(f64, usize)> = donors.iter().map(|&j| {
        let u: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        ((usage[j] as f64).ln() - (-(u.ln())).ln(), j)
    }).collect();
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut pairs = Vec::with_capacity(nsplit);
    for (t, &(_, src)) in keyed.iter().take(nsplit).enumerate() {
        let dst = unused[t];
        let half = usage[src] / 2.0;
        usage[src] = half;
        usage[dst] = half;
        pairs.push((src, dst));
    }
    pairs
}

fn next_up(v: f32) -> f32 {
    if v.is_nan() || v == f32::INFINITY {
        return v;
    }
    if v == 0.0 {
        return f32::from_bits(1);
    }
    let b = v.to_bits();
    f32::from_bits(if v > 0.0 { b + 1 } else { b - 1 })
}

// ---------------------------------------------------------------- quality measures

/// Mean squared distance to the nearest centroid.
pub fn objective(x: &Mat, c: &[f32], k: usize) -> f64 {
    let (_, d) = nearest(x, None, c, k, Metric::L2);
    d.iter().map(|&v| v as f64).sum::<f64>() / x.n as f64
}

pub fn balance(assign: &[u32], k: usize) -> serde_json::Value {
    let mut sizes = vec![0f64; k];
    for &a in assign {
        sizes[a as usize] += 1.0;
    }
    let total: f64 = sizes.iter().sum();
    let mean = total / k as f64;
    let max = sizes.iter().cloned().fold(0.0, f64::max);
    let var = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / k as f64;
    let ent: f64 = sizes.iter().filter(|&&s| s > 0.0).map(|s| { let p = s / total; -p * p.ln() }).sum();
    json!({"empty": sizes.iter().filter(|&&s| s == 0.0).count(), "max_over_mean": max / mean,
           "cv": var.sqrt() / mean, "norm_entropy": if k > 1 { ent / (k as f64).ln() } else { 1.0 },
           "imbalance_factor": k as f64 * sizes.iter().map(|s| s * s).sum::<f64>() / (total * total)})
}

/// Adjusted Rand index between two labelings of the same items.
pub fn adjusted_rand(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len());
    use std::collections::HashMap;
    let mut table: HashMap<(u32, u32), f64> = HashMap::new();
    let (mut ra, mut rb): (HashMap<u32, f64>, HashMap<u32, f64>) = (HashMap::new(), HashMap::new());
    for (&x, &y) in a.iter().zip(b) {
        *table.entry((x, y)).or_default() += 1.0;
        *ra.entry(x).or_default() += 1.0;
        *rb.entry(y).or_default() += 1.0;
    }
    let comb = |v: f64| v * (v - 1.0) / 2.0;
    let index: f64 = table.values().map(|&v| comb(v)).sum();
    let rows: f64 = ra.values().map(|&v| comb(v)).sum();
    let cols: f64 = rb.values().map(|&v| comb(v)).sum();
    let total = comb(a.len() as f64);
    let expected = rows * cols / total;
    let maximum = (rows + cols) / 2.0;
    if maximum == expected { 1.0 } else { (index - expected) / (maximum - expected) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Well-separated Gaussian blobs on the unit sphere.
    pub fn blobs(k: usize, per: usize, d: usize, seed: u64) -> (Mat, Vec<u32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centres: Vec<f32> = (0..k * d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let (mut data, mut truth) = (vec![], vec![]);
        for j in 0..k {
            for _ in 0..per {
                for t in 0..d {
                    data.push(centres[j * d + t] * 4.0 + (rng.gen::<f32>() - 0.5) * 0.3);
                }
                truth.push(j as u32);
            }
        }
        let mut m = Mat { data, n: k * per, d };
        m.normalise();
        (m, truth)
    }

    #[test]
    fn recipe_matches_reference_values() {
        // quality = throughput / 8 before power-of-two rounding, so the rounded batches differ by exactly 8x
        assert_eq!(epoch_rows(10, 5.0), vec![10; 5]);
        assert_eq!(epoch_rows(10, 1.5), vec![5, 10]);
        let (bt, _) = recipe(2_383_196, 4096, 5.0, BatchLaw::Throughput);
        let (bq, _) = recipe(2_383_196, 4096, 5.0, BatchLaw::Quality);
        assert!(bt.is_power_of_two() && bq.is_power_of_two() && bt == 8 * bq, "{bt} {bq}");
        assert_eq!(resolve_law(BatchLaw::Auto, 768), BatchLaw::Throughput);
    }

    #[test]
    fn both_backends_recover_blobs() {
        let (x, truth) = blobs(8, 400, 16, 7);
        for b in [Backend::Lloyd { niter: 25, max_points_per_centroid: None }, Backend::emeans_quality(),
                  Backend::Emeans { nepochs: 5.0, batch_law: BatchLaw::Throughput }] {
            // random init has local minima on separated blobs; best objective over restarts must be the truth
            let best = (0..30u64).map(|s| {
                let c = fit(&x, 8, &b, 100 + s).unwrap();
                (objective(&x, &c.centroids, 8), adjusted_rand(&c.assign, &truth))
            }).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
            assert!(best.1 > 0.99, "{b:?}: ARI {}", best.1);
        }
    }

    #[test]
    fn ari_basics() {
        assert!((adjusted_rand(&[0, 0, 1, 1], &[5, 5, 9, 9]) - 1.0).abs() < 1e-12);
        assert!(adjusted_rand(&[0, 0, 1, 1], &[0, 1, 0, 1]) < 0.0);
    }
}
