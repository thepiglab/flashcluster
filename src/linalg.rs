//! Dense kernels: A·Bᵀ via the `gemm` crate (runtime AVX2/AVX-512/NEON dispatch), nearest centroid, top-k.
use crate::io::Mat;
use rayon::prelude::*;
use std::cmp::Ordering;

/// out (m×n, row-major) = a (m×d) · bᵀ (b is n×d). `parallel` lets gemm use all rayon threads (use it only
/// outside an already-parallel loop).
pub fn gemm_abt(a: &[f32], m: usize, b: &[f32], n: usize, d: usize, out: &mut [f32], parallel: bool) {
    debug_assert!(a.len() >= m * d && b.len() >= n * d && out.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    let par = if parallel { gemm::Parallelism::Rayon(rayon::current_num_threads()) } else { gemm::Parallelism::None };
    unsafe {
        gemm::gemm(
            m, n, d,
            out.as_mut_ptr(), 1, n as isize, false,
            a.as_ptr(), 1, d as isize,
            b.as_ptr(), d as isize, 1,
            0.0, 1.0, false, false, false, par,
        );
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // eight independent accumulators so the compiler vectorises the loop
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

pub fn normalise_rows(x: &mut [f32], d: usize) {
    x.par_chunks_mut(d).for_each(|r| {
        let n = dot(r, r).sqrt().max(1e-12);
        r.iter_mut().for_each(|v| *v /= n);
    });
}

pub fn sq_norms(x: &[f32], d: usize) -> Vec<f32> {
    x.par_chunks(d).map(|r| dot(r, r)).collect()
}

#[derive(Clone, Copy, PartialEq)]
pub enum Metric {
    L2, // squared L2, smaller is better
    Ip, // inner product, larger is better
}

/// Nearest centroid for every row of x (or for rows `idx` of x). Returns (label, score) where score is the
/// squared L2 distance (clamped at 0) or the inner product.
pub fn nearest(x: &Mat, idx: Option<&[u32]>, c: &[f32], k: usize, metric: Metric) -> (Vec<u32>, Vec<f32>) {
    let d = x.d;
    let n = idx.map_or(x.n, |i| i.len());
    let c2 = if metric == Metric::L2 { sq_norms(c, d) } else { vec![] };
    let mut lab = vec![0u32; n];
    let mut sc = vec![0f32; n];
    const B: usize = 256;
    lab.par_chunks_mut(B).zip(sc.par_chunks_mut(B)).enumerate().for_each(|(bi, (lb, sb))| {
        let r0 = bi * B;
        let m = lb.len();
        let owned;
        let xb: &[f32] = match idx {
            None => &x.data[r0 * d..(r0 + m) * d],
            Some(ix) => {
                let mut g = Vec::with_capacity(m * d);
                for &i in &ix[r0..r0 + m] {
                    g.extend_from_slice(x.row(i as usize));
                }
                owned = g;
                &owned
            }
        };
        let mut s = vec![0f32; m * k];
        gemm_abt(xb, m, c, k, d, &mut s, false);
        for r in 0..m {
            let row = &s[r * k..(r + 1) * k];
            match metric {
                Metric::L2 => {
                    let (mut bj, mut bv) = (0usize, f32::INFINITY);
                    for j in 0..k {
                        let v = c2[j] - 2.0 * row[j];
                        if v < bv {
                            bv = v;
                            bj = j;
                        }
                    }
                    let xr = &xb[r * d..(r + 1) * d];
                    lb[r] = bj as u32;
                    sb[r] = (bv + dot(xr, xr)).max(0.0);
                }
                Metric::Ip => {
                    let (mut bj, mut bv) = (0usize, f32::NEG_INFINITY);
                    for j in 0..k {
                        if row[j] > bv {
                            bv = row[j];
                            bj = j;
                        }
                    }
                    lb[r] = bj as u32;
                    sb[r] = bv;
                }
            }
        }
    });
    (lab, sc)
}

/// Bounded best-k collector for inner-product scores (larger is better); ties broken by smaller id.
#[derive(Clone)]
pub struct TopK {
    pub k: usize,
    items: Vec<(f32, u32)>, // kept unsorted until full, then as a min-heap by score
}

fn worse(a: &(f32, u32), b: &(f32, u32)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 > b.1)
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK { k, items: Vec::with_capacity(k + 1) }
    }
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.items.len() < self.k { f32::NEG_INFINITY } else { self.items[0].0 }
    }
    #[inline]
    pub fn push(&mut self, s: f32, id: u32) {
        if self.k == 0 {
            return;
        }
        if self.items.len() < self.k {
            self.items.push((s, id));
            if self.items.len() == self.k {
                self.heapify();
            }
        } else if worse(&self.items[0], &(s, id)) {
            self.items[0] = (s, id);
            self.sift(0);
        }
    }
    fn heapify(&mut self) {
        for i in (0..self.items.len() / 2).rev() {
            self.sift(i);
        }
    }
    fn sift(&mut self, mut i: usize) {
        let n = self.items.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < n && worse(&self.items[l], &self.items[m]) {
                m = l;
            }
            if r < n && worse(&self.items[r], &self.items[m]) {
                m = r;
            }
            if m == i {
                return;
            }
            self.items.swap(i, m);
            i = m;
        }
    }
    /// Best first.
    pub fn into_sorted(mut self) -> Vec<(f32, u32)> {
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal).then(a.1.cmp(&b.1)));
        self.items
    }
}

/// Exact inner-product top-`depth` of each query row against all rows of `x` (blocked GEMM).
pub fn flat_search(x: &Mat, q: &Mat, depth: usize) -> Vec<Vec<(f32, u32)>> {
    let d = x.d;
    let mut tops: Vec<TopK> = (0..q.n).map(|_| TopK::new(depth)).collect();
    const XB: usize = 16384;
    let mut s = vec![0f32; q.n.min(4096) * XB];
    for q0 in (0..q.n).step_by(4096) {
        let qm = (q.n - q0).min(4096);
        let qd = &q.data[q0 * d..(q0 + qm) * d];
        for x0 in (0..x.n).step_by(XB) {
            let xm = (x.n - x0).min(XB);
            gemm_abt(qd, qm, &x.data[x0 * d..(x0 + xm) * d], xm, d, &mut s[..qm * xm], true);
            tops[q0..q0 + qm].par_iter_mut().enumerate().for_each(|(r, t)| {
                let row = &s[r * xm..(r + 1) * xm];
                for (j, &v) in row.iter().enumerate() {
                    if v > t.threshold() {
                        t.push(v, (x0 + j) as u32);
                    }
                }
            });
        }
    }
    tops.into_iter().map(TopK::into_sorted).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_naive() {
        let (m, n, d) = (5, 7, 13);
        let a: Vec<f32> = (0..m * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut o = vec![0f32; m * n];
        gemm_abt(&a, m, &b, n, d, &mut o, false);
        for i in 0..m {
            for j in 0..n {
                let e = dot(&a[i * d..(i + 1) * d], &b[j * d..(j + 1) * d]);
                assert!((o[i * n + j] - e).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn topk_keeps_best() {
        let mut t = TopK::new(3);
        for (i, s) in [0.1, 0.9, 0.5, 0.7, 0.2, 0.95].iter().enumerate() {
            t.push(*s, i as u32);
        }
        let ids: Vec<u32> = t.into_sorted().iter().map(|x| x.1).collect();
        assert_eq!(ids, vec![5, 1, 3]);
    }
}
