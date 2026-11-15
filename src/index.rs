//! IVF index (inner product) over f16 lists, built from external centroids.
//!
//! Design (why it beats IVF-Flat in FAISS on our data):
//! * lists hold the vectors as f16, contiguous per list: exact for our embeddings (stored f16 upstream), half the
//!   bytes of an f32 IVF-Flat, and IVF scanning is memory-bandwidth bound;
//! * explicit AVX-512 / AVX2+F16C kernels score 8 rows per call with f32 accumulation (`simd`);
//! * the index is built in place (labels, then an in-place row permutation): peak memory ≈ one copy of the data;
//! * batch search is list-major: each probed list is streamed once for all queries that probe it (the list tile
//!   stays in cache while every query of that list is scored), then per-query candidates are merged;
//! * single-query search scans its probed lists directly with a fixed-size top-k and a threshold test.
use crate::io::Mat;
use crate::kmeans::group_by_label;
use crate::linalg::{gemm_abt, normalise_rows, TopK};
use crate::quant::{self, QuantQuery, QuantRows};
use crate::simd::{dot_rows_f16, dot_rows_f32, R};
use half::f16;
use rayon::prelude::*;

pub struct Ivf {
    pub d: usize,
    pub k: usize,
    cent: Vec<f32>,          // normalised centroids, k×d (f32, as FAISS's flat quantizer)
    offsets: Vec<usize>,     // list j = rows offsets[j]..offsets[j+1]
    ids: Vec<u32>,           // original row id per stored row
    vecs: Vec<f16>,          // stored rows, list-contiguous (empty once quantised)
    q8: Option<QuantRows>,   // int8 rows (per-vector scale), list-contiguous, after `quantize_int8`
}

/// Rows of a list tile kept hot while every query of that list is scored (64 × 768 × 2 B = 96 KiB).
const TILE: usize = 64;
/// Lists probed by at least this many queries of a batch are scored with one f32 GEMM per converted tile.
const GEMM_MIN_QUERIES: usize = 4;
fn gemm_min_queries() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FLASHCLUSTER_GEMM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(GEMM_MIN_QUERIES))
}
/// Rows converted per GEMM tile (128 × 768 × 4 B = 384 KiB, L2-resident).
const GEMM_TILE: usize = 128;

impl Ivf {
    /// Build from an f16 pool (n×d, consumed and permuted in place).
    pub fn build_f16(mut pool: Vec<f16>, d: usize, centroids: &[f32], k: usize) -> Ivf {
        let n = pool.len() / d;
        let mut cent = centroids.to_vec();
        normalise_rows(&mut cent, d);
        // coarse assignment: blocks of rows converted to f32, GEMM against centroids, argmax
        const B: usize = 1024;
        let mut lab = vec![0u32; n];
        lab.par_chunks_mut(B).enumerate().for_each(|(bi, lb)| {
            let m = lb.len();
            let xb: Vec<f32> = pool[bi * B * d..(bi * B + m) * d].iter().map(|v| v.to_f32()).collect();
            let mut s = vec![0f32; m * k];
            gemm_abt(&xb, m, &cent, k, d, &mut s, false);
            for r in 0..m {
                let row = &s[r * k..(r + 1) * k];
                let mut best = 0usize;
                for j in 1..k {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                lb[r] = best as u32;
            }
        });
        let (order, offsets) = group_by_label(&lab, k);
        drop(lab);
        // in place: new row t = old row order[t] (cycle-following, one spare row)
        let mut done = vec![false; n];
        let mut tmp = vec![f16::ZERO; d];
        for start in 0..n {
            if done[start] || order[start] as usize == start {
                done[start] = true;
                continue;
            }
            tmp.copy_from_slice(&pool[start * d..(start + 1) * d]);
            let mut j = start;
            loop {
                let src = order[j] as usize;
                done[j] = true;
                if src == start {
                    pool[j * d..(j + 1) * d].copy_from_slice(&tmp);
                    break;
                }
                pool.copy_within(src * d..(src + 1) * d, j * d);
                j = src;
            }
        }
        Ivf { d, k, cent, offsets, ids: order, vecs: pool, q8: None }
    }

    /// Build from an f32 matrix (rounded to f16 storage).
    pub fn build(x: &Mat, centroids: &[f32], k: usize) -> Ivf {
        let pool: Vec<f16> = x.data.par_iter().map(|&v| f16::from_f32(v)).collect();
        Ivf::build_f16(pool, x.d, centroids, k)
    }

    pub fn list_size(&self, j: usize) -> usize {
        self.offsets[j + 1] - self.offsets[j]
    }

    pub fn memory_bytes(&self) -> usize {
        let q8 = self.q8.as_ref().map_or(0, |q| q.q.len() + q.scale.len() * 8);
        self.vecs.len() * 2 + q8 + self.ids.len() * 4 + self.cent.len() * 4 + self.offsets.len() * 8
    }

    /// Replace the f16 lists by int8 lists with a per-vector scale (half the bytes; approximate scores).
    pub fn quantize_int8(&mut self) {
        self.q8 = Some(quant::quantize_rows_f16_in_place(std::mem::take(&mut self.vecs), self.d));
    }

    pub fn is_int8(&self) -> bool {
        self.q8.is_some()
    }

    #[inline]
    fn scan_i8(&self, q: &QuantQuery, lo: usize, hi: usize, top: &mut TopK) {
        let (d, r) = (self.d, self.q8.as_ref().unwrap());
        for t in lo..hi {
            let v = quant::dot(q, &r.q[t * d..(t + 1) * d], r.rsum[t]) as f32 * q.scale * r.scale[t];
            if v > top.threshold() {
                top.push(v, self.ids[t]);
            }
        }
    }

    /// Score rows [lo, hi) of storage against q into `top`.
    #[inline]
    fn scan(&self, q: &[f32], lo: usize, hi: usize, top: &mut TopK) {
        let d = self.d;
        let mut s = [0f32; R];
        let mut t = lo;
        while t < hi {
            let m = (hi - t).min(R);
            dot_rows_f16(q, &self.vecs[t * d..(t + m) * d], m, &mut s);
            for (r, &v) in s.iter().enumerate().take(m) {
                if v > top.threshold() {
                    top.push(v, self.ids[t + r]);
                }
            }
            t += m;
        }
    }

    fn probes_one(&self, q: &[f32], nprobe: usize) -> TopK {
        let (d, k) = (self.d, self.k);
        let mut probe = TopK::new(nprobe.min(k));
        let mut s = [0f32; R];
        let mut j = 0;
        while j < k {
            let m = (k - j).min(R);
            dot_rows_f32(q, &self.cent[j * d..(j + m) * d], m, &mut s);
            for (r, &v) in s.iter().enumerate().take(m) {
                if v > probe.threshold() {
                    probe.push(v, (j + r) as u32);
                }
            }
            j += m;
        }
        probe
    }

    /// One query (latency path, runs on the calling thread).
    pub fn search_one(&self, q: &[f32], nprobe: usize, depth: usize) -> Vec<(f32, u32)> {
        let mut top = TopK::new(depth);
        let qq = self.q8.as_ref().map(|_| quant::quantize_query(q));
        for (_, j) in self.probes_one(q, nprobe).into_sorted() {
            let j = j as usize;
            match &qq {
                Some(qq) => self.scan_i8(qq, self.offsets[j], self.offsets[j + 1], &mut top),
                None => self.scan(q, self.offsets[j], self.offsets[j + 1], &mut top),
            }
        }
        top.into_sorted()
    }

    /// Many queries (throughput path, list-major, uses the current rayon pool).
    pub fn search(&self, q: &Mat, nprobe: usize, depth: usize) -> Vec<Vec<(f32, u32)>> {
        let (d, k, nq) = (self.d, self.k, q.n);
        let nprobe = nprobe.min(k);
        // coarse: GEMM, then top-nprobe per query
        let mut qc = vec![0f32; nq * k];
        gemm_abt(&q.data, nq, &self.cent, k, d, &mut qc, true);
        let probes: Vec<Vec<u32>> = qc.par_chunks(k).map(|row| {
            let mut p = TopK::new(nprobe);
            for (j, &v) in row.iter().enumerate() {
                if v > p.threshold() {
                    p.push(v, j as u32);
                }
            }
            p.into_sorted().into_iter().map(|x| x.1).collect()
        }).collect();
        drop(qc);
        // invert: queries per list
        let mut cnt = vec![0usize; k + 1];
        for p in &probes {
            for &j in p {
                cnt[j as usize + 1] += 1;
            }
        }
        for j in 0..k {
            cnt[j + 1] += cnt[j];
        }
        let mut fill = cnt.clone();
        let mut lq = vec![0u32; cnt[k]];
        for (qi, p) in probes.iter().enumerate() {
            for &j in p {
                lq[fill[j as usize]] = qi as u32;
                fill[j as usize] += 1;
            }
        }
        let mut lists: Vec<usize> = (0..k).filter(|&j| cnt[j + 1] > cnt[j] && self.list_size(j) > 0).collect();
        lists.sort_by_key(|&j| std::cmp::Reverse((cnt[j + 1] - cnt[j]) * self.list_size(j)));
        let qq8: Option<Vec<QuantQuery>> = self.q8.as_ref().map(|_| (0..nq).into_par_iter().map(|i| quant::quantize_query(q.row(i))).collect());
        // list-major scan: per list, every probing query against a cached tile of the list
        let partial: Vec<Vec<(u32, Vec<(f32, u32)>)>> = lists.par_iter().map(|&j| {
            let qs = &lq[cnt[j]..cnt[j + 1]];
            let mut tops: Vec<TopK> = (0..qs.len()).map(|_| TopK::new(depth)).collect();
            let (lo, hi) = (self.offsets[j], self.offsets[j + 1]);
            if let Some(qqs) = &qq8 {
                // groups of 4 queries share each row load (dot4); remainder one by one
                let (r8, d) = (self.q8.as_ref().unwrap(), self.d);
                let mut t0 = lo;
                while t0 < hi {
                    let t1 = (t0 + TILE * 2).min(hi);
                    let mut g = 0;
                    while g + 4 <= qs.len() {
                        let qq = [&qqs[qs[g] as usize], &qqs[qs[g + 1] as usize], &qqs[qs[g + 2] as usize], &qqs[qs[g + 3] as usize]];
                        for t in t0..t1 {
                            let v = quant::dot4(qq, &r8.q[t * d..(t + 1) * d], r8.rsum[t]);
                            for e in 0..4 {
                                let sc = v[e] as f32 * qq[e].scale * r8.scale[t];
                                if sc > tops[g + e].threshold() {
                                    tops[g + e].push(sc, self.ids[t]);
                                }
                            }
                        }
                        g += 4;
                    }
                    for (qi, top) in qs[g..].iter().zip(tops[g..].iter_mut()) {
                        self.scan_i8(&qqs[*qi as usize], t0, t1, top);
                    }
                    t0 = t1;
                }
            } else if qs.len() >= gemm_min_queries() {
                // many queries on this list: f16 tile -> f32 once, then one GEMM for all of them
                let m = qs.len();
                let mut qb = Vec::with_capacity(m * d);
                for &qi in qs {
                    qb.extend_from_slice(q.row(qi as usize));
                }
                let mut tile = vec![0f32; GEMM_TILE * d];
                let mut sc = vec![0f32; m * GEMM_TILE];
                let mut t0 = lo;
                while t0 < hi {
                    let nt = (hi - t0).min(GEMM_TILE);
                    for (o, v) in tile[..nt * d].iter_mut().zip(&self.vecs[t0 * d..(t0 + nt) * d]) {
                        *o = v.to_f32();
                    }
                    gemm_abt(&qb, m, &tile[..nt * d], nt, d, &mut sc[..m * nt], false);
                    for (r, top) in tops.iter_mut().enumerate() {
                        for (c, &v) in sc[r * nt..(r + 1) * nt].iter().enumerate() {
                            if v > top.threshold() {
                                top.push(v, self.ids[t0 + c]);
                            }
                        }
                    }
                    t0 += nt;
                }
            } else {
                let mut t0 = lo;
                while t0 < hi {
                    let t1 = (t0 + TILE).min(hi);
                    for (qi, top) in qs.iter().zip(tops.iter_mut()) {
                        self.scan(q.row(*qi as usize), t0, t1, top);
                    }
                    t0 = t1;
                }
            }
            qs.iter().zip(tops).map(|(&qi, t)| (qi, t.into_sorted())).collect()
        }).collect();
        // merge per query
        let mut per: Vec<Vec<Vec<(f32, u32)>>> = (0..nq).map(|_| Vec::with_capacity(nprobe)).collect();
        for part in partial {
            for (qi, items) in part {
                per[qi as usize].push(items);
            }
        }
        per.into_par_iter().map(|parts| {
            let mut top = TopK::new(depth);
            for p in parts {
                for (s, id) in p {
                    if s > top.threshold() {
                        top.push(s, id);
                    }
                }
            }
            top.into_sorted()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::{fit, Backend};
    use crate::linalg::flat_search;

    fn data() -> Mat {
        let mut x = Mat { data: (0..3000 * 32).map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5)).collect(), n: 3000, d: 32 };
        x.normalise();
        // round through f16 so exact search sees the stored values
        x.data.iter_mut().for_each(|v| *v = f16::from_f32(*v).to_f32());
        x
    }

    #[test]
    fn full_probe_equals_exact_and_paths_agree() {
        let x = data();
        let q = x.gather(&(0..60).collect::<Vec<u32>>());
        let c = fit(&x, 20, &Backend::emeans_quality(), 1).unwrap();
        let ivf = Ivf::build(&x, &c.centroids, 20);
        let exact = flat_search(&x, &q, 10);
        let batch = ivf.search(&q, 20, 10);
        for r in 0..q.n {
            let ids = |v: &Vec<(f32, u32)>| v.iter().map(|t| t.1).collect::<Vec<_>>();
            assert_eq!(ids(&batch[r]), ids(&exact[r]), "query {r}");
            for p in [1, 3, 20] {
                assert_eq!(ids(&ivf.search_one(q.row(r), p, 10)), ids(&ivf.search(&q.gather(&[r as u32]), p, 10)[0]));
            }
        }
    }

    #[test]
    fn in_place_build_keeps_rows() {
        let x = data();
        let c = fit(&x, 7, &Backend::emeans_quality(), 2).unwrap();
        let ivf = Ivf::build(&x, &c.centroids, 7);
        for t in (0..x.n).step_by(97) {
            let orig = ivf.ids[t] as usize;
            for e in 0..x.d {
                assert_eq!(ivf.vecs[t * x.d + e].to_f32(), x.data[orig * x.d + e]);
            }
        }
    }
}
