//! Nested hierarchies over a flat clustering, and agreement measures between runs.
//!
//! * `weighted_kmeans`: k-means++ and Lloyd on weighted points (here: leaf centroids weighted by leaf size). For leaf
//!   centroids c_l (the means of their members) and a coarse centroid m, the members' squared error splits exactly
//!   into Σ_l w_l·|c_l − m|² + (within-leaf error), so this minimises the members' objective subject to every leaf
//!   staying whole: the tree nests exactly and coarse centroids are the means of their members.
//! * `build_levels`: repeated coarsening, finest first (e.g. leaves → 1024 → 64). Measured worse and far less
//!   stable across seeds than flat e-means at each k (Lloyd on leaf centroids stalls in local minima), so `gpu-fit`
//!   defaults to flat fits nested by majority (`nest_majority`, `nested_centroids`).
//! * `node_stability`: per node of a reference run, mean over other runs of the best Jaccard overlap with any node.
//! * `prefix_centroids`: centroids in the first `p` dimensions (Matryoshka prefixes) for a given partition.
use crate::linalg::{dot, gemm_abt};
use half::f16;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Weighted {
    pub centroids: Vec<f32>,
    pub labels: Vec<u32>,
    /// Σ w·|x − c|² / Σ w
    pub objective: f64,
    pub iterations: usize,
}

fn sq_norm(r: &[f32]) -> f32 {
    dot(r, r)
}

/// Nearest centroid (squared L2) for every point; returns (labels, squared distances).
fn assign(x: &[f32], n: usize, c: &[f32], k: usize, d: usize) -> (Vec<u32>, Vec<f32>) {
    let cn: Vec<f32> = c.par_chunks(d).map(sq_norm).collect();
    const B: usize = 4096;
    let mut lab = vec![0u32; n];
    let mut dist = vec![0f32; n];
    lab.par_chunks_mut(B).zip(dist.par_chunks_mut(B)).enumerate().for_each(|(bi, (lo, dd))| {
        let s0 = bi * B;
        let m = lo.len();
        let mut ip = vec![0f32; m * k];
        gemm_abt(&x[s0 * d..(s0 + m) * d], m, c, k, d, &mut ip, false);
        for r in 0..m {
            let xn = sq_norm(&x[(s0 + r) * d..(s0 + r + 1) * d]);
            let (mut best, mut bj) = (f32::INFINITY, 0u32);
            for j in 0..k {
                let v = cn[j] - 2.0 * ip[r * k + j];
                if v < best {
                    best = v;
                    bj = j as u32;
                }
            }
            lo[r] = bj;
            dd[r] = (best + xn).max(0.0);
        }
    });
    (lab, dist)
}

/// Weighted k-means: weighted k-means++ seeding, then Lloyd until no label changes or `niter`; empty clusters are
/// re-seeded at the point with the largest weighted error. Best objective over `restarts`.
pub fn weighted_kmeans(x: &[f32], w: &[f64], d: usize, k: usize, niter: usize, restarts: usize, seed: u64) -> Weighted {
    let n = w.len();
    assert!(k >= 1 && k <= n && x.len() == n * d);
    let tw: f64 = w.iter().sum();
    let mut best: Option<Weighted> = None;
    for r in 0..restarts.max(1) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(r as u64 * 0x9E37_79B9));
        // k-means++: first centre ∝ w, then ∝ w·D²
        let pick = |p: &[f64], rng: &mut ChaCha8Rng| -> usize {
            let tot: f64 = p.iter().sum();
            if tot <= 0.0 {
                return rng.gen_range(0..p.len());
            }
            let mut t = rng.gen::<f64>() * tot;
            for (i, &v) in p.iter().enumerate() {
                t -= v;
                if t <= 0.0 {
                    return i;
                }
            }
            p.len() - 1
        };
        let mut c = Vec::with_capacity(k * d);
        let first = pick(w, &mut rng);
        c.extend_from_slice(&x[first * d..(first + 1) * d]);
        let mut d2: Vec<f64> = (0..n).into_par_iter().map(|i| {
            let (a, b) = (&x[i * d..(i + 1) * d], &c[..d]);
            a.iter().zip(b).map(|(p, q)| ((p - q) * (p - q)) as f64).sum::<f64>()
        }).collect();
        for _ in 1..k {
            let p: Vec<f64> = d2.iter().zip(w).map(|(a, b)| a * b).collect();
            let i = pick(&p, &mut rng);
            let new = x[i * d..(i + 1) * d].to_vec();
            d2.par_iter_mut().enumerate().for_each(|(t, v)| {
                let a = &x[t * d..(t + 1) * d];
                let e: f64 = a.iter().zip(&new).map(|(p, q)| ((p - q) * (p - q)) as f64).sum();
                if e < *v {
                    *v = e;
                }
            });
            c.extend_from_slice(&new);
        }
        let mut labels = vec![u32::MAX; n];
        let mut it = 0;
        let mut dist;
        loop {
            let (lab, dd) = assign(x, n, &c, k, d);
            dist = dd;
            let changed = lab.iter().zip(&labels).filter(|(a, b)| a != b).count();
            labels = lab;
            it += 1;
            if changed == 0 || it > niter {
                break;
            }
            let mut sums = vec![0f64; k * d];
            let mut mass = vec![0f64; k];
            for i in 0..n {
                let j = labels[i] as usize;
                mass[j] += w[i];
                for t in 0..d {
                    sums[j * d + t] += w[i] * x[i * d + t] as f64;
                }
            }
            for j in 0..k {
                if mass[j] > 0.0 {
                    for t in 0..d {
                        c[j * d + t] = (sums[j * d + t] / mass[j]) as f32;
                    }
                }
            }
            // re-seed empty clusters at the worst-served points (each point used at most once)
            let empty: Vec<usize> = (0..k).filter(|&j| mass[j] == 0.0).collect();
            if !empty.is_empty() {
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| (w[b] * dist[b] as f64).partial_cmp(&(w[a] * dist[a] as f64)).unwrap());
                for (j, &i) in empty.iter().zip(&order) {
                    c[j * d..(j + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
                }
            }
        }
        let objective = dist.iter().zip(w).map(|(a, b)| *a as f64 * b).sum::<f64>() / tw;
        if best.as_ref().map_or(true, |b| objective < b.objective) {
            best = Some(Weighted { centroids: c, labels, objective, iterations: it });
        }
    }
    best.unwrap()
}

/// Sizes of each label (0..k).
pub fn sizes(labels: &[u32], k: usize) -> Vec<u64> {
    let mut s = vec![0u64; k];
    for &l in labels {
        s[l as usize] += 1;
    }
    s
}

pub struct Level {
    pub k: usize,
    pub centroids: Vec<f32>,
    /// node of the next finer level → node of this level
    pub parent_of: Vec<u32>,
    pub objective: f64,
    /// this level's node for every row (empty = follow `parent_of` from the finer level's rows)
    pub assign: Vec<u32>,
}

/// Coarsen a leaf clustering into nested levels, finest first: `ks` = e.g. [1024, 64]. Leaves (or nodes) with zero
/// members get weight 0 and are attached to their nearest coarse centroid.
pub fn build_levels(leaf_c: &[f32], leaf_sizes: &[u64], d: usize, ks: &[usize], seed: u64) -> Vec<Level> {
    let mut out: Vec<Level> = vec![];
    let (mut c, mut w): (Vec<f32>, Vec<f64>) = (leaf_c.to_vec(), leaf_sizes.iter().map(|&s| s as f64).collect());
    for (li, &k) in ks.iter().enumerate() {
        let r = weighted_kmeans(&c, &w, d, k, 100, 3, seed + li as u64);
        let mut nw = vec![0f64; k];
        for (i, &l) in r.labels.iter().enumerate() {
            nw[l as usize] += w[i];
        }
        out.push(Level { k, centroids: r.centroids.clone(), parent_of: r.labels, objective: r.objective, assign: vec![] });
        c = r.centroids;
        w = nw;
    }
    out
}

/// Nest a finer partition under a coarser one by majority: each child node goes to the parent node holding most of
/// its rows (ties: lowest parent id). Returns (parent_of child node, share of rows whose own parent label agrees).
pub fn nest_majority(child: &[u32], k_child: usize, parent: &[u32]) -> (Vec<u32>, f64) {
    let mut pair: HashMap<(u32, u32), u64> = HashMap::with_capacity(4 * k_child);
    for (&c, &p) in child.iter().zip(parent) {
        *pair.entry((c, p)).or_default() += 1;
    }
    let mut best = vec![(0u64, u32::MAX); k_child];
    for (&(c, p), &n) in &pair {
        let b = &mut best[c as usize];
        if n > b.0 || (n == b.0 && p < b.1) {
            *b = (n, p);
        }
    }
    let agree: u64 = best.iter().map(|b| b.0).sum();
    // empty child nodes: attach to parent 0 (they have no rows)
    (best.iter().map(|b| if b.1 == u32::MAX { 0 } else { b.1 }).collect(), agree as f64 / child.len().max(1) as f64)
}

/// Centroids of parent nodes as size-weighted means of their children's centroids.
pub fn nested_centroids(child_c: &[f32], child_sizes: &[u64], parent_of: &[u32], k_parent: usize, d: usize) -> Vec<f32> {
    let mut sum = vec![0f64; k_parent * d];
    let mut w = vec![0f64; k_parent];
    for (j, &p) in parent_of.iter().enumerate() {
        let s = child_sizes[j] as f64;
        w[p as usize] += s;
        for t in 0..d {
            sum[p as usize * d + t] += s * child_c[j * d + t] as f64;
        }
    }
    (0..k_parent * d).map(|i| if w[i / d] > 0.0 { (sum[i] / w[i / d]) as f32 } else { 0.0 }).collect()
}

/// Per reference node: mean over `others` of the best Jaccard overlap with any node of that run.
pub fn node_stability(r: &[u32], others: &[&[u32]]) -> Vec<f64> {
    let kr = *r.iter().max().unwrap_or(&0) as usize + 1;
    let sr = sizes(r, kr);
    let mut out = vec![0f64; kr];
    for o in others {
        let ko = *o.iter().max().unwrap_or(&0) as usize + 1;
        let so = sizes(o, ko);
        let mut pair: HashMap<(u32, u32), u64> = HashMap::with_capacity(4 * kr);
        for (&a, &b) in r.iter().zip(o.iter()) {
            *pair.entry((a, b)).or_default() += 1;
        }
        let mut best = vec![0f64; kr];
        for (&(a, b), &inter) in &pair {
            let j = inter as f64 / (sr[a as usize] + so[b as usize] - inter) as f64;
            if j > best[a as usize] {
                best[a as usize] = j;
            }
        }
        out.iter_mut().zip(best).for_each(|(x, b)| *x += b);
    }
    out.iter().map(|x| x / others.len().max(1) as f64).collect()
}

/// Centroids of a partition in the first `p` dimensions: each row's prefix is L2-normalised, then averaged per
/// label and the mean re-normalised (what a reader of the p-dim prefix vectors would compute).
pub fn prefix_centroids(x: &[f16], d: usize, labels: &[u32], k: usize, p: usize) -> Vec<f32> {
    assert!(p <= d);
    let sums = labels.par_iter().enumerate().fold(|| vec![0f64; k * p], |mut acc, (i, &l)| {
        let row: Vec<f32> = x[i * d..i * d + p].iter().map(|v| v.to_f32()).collect();
        let nrm = dot(&row, &row).sqrt().max(1e-12);
        let base = l as usize * p;
        for t in 0..p {
            acc[base + t] += (row[t] / nrm) as f64;
        }
        acc
    }).reduce(|| vec![0f64; k * p], |mut a, b| {
        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        a
    });
    let mut c: Vec<f32> = sums.into_iter().map(|v| v as f32).collect();
    c.par_chunks_mut(p).for_each(|r| {
        let n = dot(r, r).sqrt();
        if n > 0.0 {
            r.iter_mut().for_each(|v| *v /= n);
        }
    });
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_kmeans_respects_weights() {
        // two far groups; a heavy point drags its group's centre
        let x = vec![0.0, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0];
        let w = vec![1.0, 3.0, 1.0, 1.0];
        let r = weighted_kmeans(&x, &w, 2, 2, 50, 5, 1);
        assert_eq!(r.labels[0], r.labels[1]);
        assert_ne!(r.labels[0], r.labels[2]);
        let j = r.labels[0] as usize;
        assert!((r.centroids[j * 2] - 0.75).abs() < 1e-6, "{:?}", r.centroids);
    }

    #[test]
    fn levels_nest_and_cover() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (n, d) = (200, 8);
        let x: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let s: Vec<u64> = (0..n).map(|i| 1 + (i % 7) as u64).collect();
        let lv = build_levels(&x, &s, d, &[20, 4], 9);
        assert_eq!(lv[0].parent_of.len(), n);
        assert_eq!(lv[1].parent_of.len(), 20);
        assert!(lv[1].parent_of.iter().all(|&p| p < 4));
    }

    #[test]
    fn stability_is_one_for_relabelled_runs() {
        let r = vec![0, 0, 1, 1, 2];
        assert_eq!(node_stability(&r, &[&[5, 5, 3, 3, 1]]), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn majority_nesting() {
        let child = [0, 0, 0, 1, 1, 2];
        let parent = [5, 5, 4, 4, 4, 0];
        let (p, agree) = nest_majority(&child, 3, &parent);
        assert_eq!(p, vec![5, 4, 0]);
        assert!((agree - 5.0 / 6.0).abs() < 1e-12);
        let c = nested_centroids(&[1.0, 3.0, 10.0], &[1, 3, 0], &[0, 0, 1], 2, 1);
        assert_eq!(c, vec![2.5, 0.0]);
    }

    #[test]
    fn prefix_centroids_normalised() {
        let x: Vec<f16> = [3.0f32, 4.0, 9.0, 0.0, 6.0, 8.0, 1.0, 1.0].iter().map(|&v| f16::from_f32(v)).collect();
        let c = prefix_centroids(&x, 4, &[0, 0], 1, 2);
        assert!((c[0] - 0.6).abs() < 1e-3 && (c[1] - 0.8).abs() < 1e-3);
    }
}
