//! Dot-product kernels for IVF scanning: one f32 query against R rows stored as f16 (IVF lists) or f32
//! (centroids). Explicit AVX-512 / AVX2+FMA+F16C kernels chosen once at runtime; portable fallback elsewhere.
//! Accumulation is f32, as in FAISS. f16 storage is exact for our vectors (the bundles store f16 embeddings).
use half::f16;
use std::sync::OnceLock;

/// Rows scored per kernel call: eight independent FMA chains hide FMA latency.
pub const R: usize = 8;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Level {
    Scalar,
    Avx2,
    Avx512,
}

pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::env::var("FLASHCLUSTER_SIMD").as_deref() == Ok("scalar") {
                return Level::Scalar;
            }
            if is_x86_feature_detected!("avx512f") && std::env::var("FLASHCLUSTER_SIMD").as_deref() != Ok("avx2") {
                return Level::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
                return Level::Avx2;
            }
        }
        Level::Scalar
    })
}

/// scores[r] = q · rows[r*d..(r+1)*d] for r < nrows (nrows <= R); rows are f16.
#[inline]
pub fn dot_rows_f16(q: &[f32], rows: &[f16], nrows: usize, scores: &mut [f32; R]) {
    let d = q.len();
    debug_assert!(nrows <= R && rows.len() >= nrows * d);
    match level() {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 if nrows == R && d % 16 == 0 => unsafe { x86::f16_avx512(q.as_ptr(), rows.as_ptr() as *const u16, d, scores) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Avx512 if nrows == R && d % 8 == 0 => unsafe { x86::f16_avx2(q.as_ptr(), rows.as_ptr() as *const u16, d, scores) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Avx512 if d % 8 == 0 => {
            for r in 0..nrows {
                scores[r] = unsafe { x86::f16_one_avx2(q.as_ptr(), rows.as_ptr().add(r * d) as *const u16, d) };
            }
        }
        _ => {
            for r in 0..nrows {
                let row = &rows[r * d..(r + 1) * d];
                let mut acc = [0f32; 8];
                for (cq, cr) in q.chunks(8).zip(row.chunks(8)) {
                    for (l, (a, b)) in cq.iter().zip(cr).enumerate() {
                        acc[l] += a * b.to_f32();
                    }
                }
                scores[r] = acc.iter().sum();
            }
        }
    }
}

/// Same for f32 rows (coarse quantizer).
#[inline]
pub fn dot_rows_f32(q: &[f32], rows: &[f32], nrows: usize, scores: &mut [f32; R]) {
    let d = q.len();
    debug_assert!(nrows <= R && rows.len() >= nrows * d);
    match level() {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Avx512 if nrows == R && d % 8 == 0 => unsafe { x86::f32_avx2(q.as_ptr(), rows.as_ptr(), d, scores) },
        _ => {
            for r in 0..nrows {
                scores[r] = crate::linalg::dot(q, &rows[r * d..(r + 1) * d]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::R;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn f16_avx2(q: *const f32, rows: *const u16, d: usize, out: &mut [f32; R]) {
        let mut acc = [_mm256_setzero_ps(); R];
        let mut i = 0;
        while i < d {
            let qv = _mm256_loadu_ps(q.add(i));
            for r in 0..R {
                let x = _mm256_cvtph_ps(_mm_loadu_si128(rows.add(r * d + i) as *const __m128i));
                acc[r] = _mm256_fmadd_ps(qv, x, acc[r]);
            }
            i += 8;
        }
        for r in 0..R {
            out[r] = hsum256(acc[r]);
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn f16_one_avx2(q: *const f32, row: *const u16, d: usize) -> f32 {
        let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 16 <= d {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(i)), _mm256_cvtph_ps(_mm_loadu_si128(row.add(i) as *const __m128i)), a);
            b = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(i + 8)), _mm256_cvtph_ps(_mm_loadu_si128(row.add(i + 8) as *const __m128i)), b);
            i += 16;
        }
        if i < d {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(i)), _mm256_cvtph_ps(_mm_loadu_si128(row.add(i) as *const __m128i)), a);
        }
        hsum256(_mm256_add_ps(a, b))
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn f16_avx512(q: *const f32, rows: *const u16, d: usize, out: &mut [f32; R]) {
        let mut acc = [_mm512_setzero_ps(); R];
        let mut i = 0;
        while i < d {
            let qv = _mm512_loadu_ps(q.add(i));
            for r in 0..R {
                let x = _mm512_cvtph_ps(_mm256_loadu_si256(rows.add(r * d + i) as *const __m256i));
                acc[r] = _mm512_fmadd_ps(qv, x, acc[r]);
            }
            i += 16;
        }
        for r in 0..R {
            out[r] = _mm512_reduce_add_ps(acc[r]);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn f32_avx2(q: *const f32, rows: *const f32, d: usize, out: &mut [f32; R]) {
        let mut acc = [_mm256_setzero_ps(); R];
        let mut i = 0;
        while i < d {
            let qv = _mm256_loadu_ps(q.add(i));
            for r in 0..R {
                acc[r] = _mm256_fmadd_ps(qv, _mm256_loadu_ps(rows.add(r * d + i)), acc[r]);
            }
            i += 8;
        }
        for r in 0..R {
            out[r] = hsum256(acc[r]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_match_reference() {
        let d = 768;
        let q: Vec<f32> = (0..d).map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5).collect();
        let rows32: Vec<f32> = (0..R * d).map(|i| ((i * 53) % 97) as f32 / 97.0 - 0.5).collect();
        let rows16: Vec<f16> = rows32.iter().map(|&v| f16::from_f32(v)).collect();
        let (mut a, mut b) = ([0f32; R], [0f32; R]);
        dot_rows_f16(&q, &rows16, R, &mut a);
        dot_rows_f32(&q, &rows32, R, &mut b);
        for r in 0..R {
            let e16: f32 = (0..d).map(|t| q[t] * rows16[r * d + t].to_f32()).sum();
            let e32: f32 = (0..d).map(|t| q[t] * rows32[r * d + t]).sum();
            assert!((a[r] - e16).abs() < 1e-3 && (b[r] - e32).abs() < 1e-3, "{r}");
        }
        let mut c = [0f32; R];
        dot_rows_f16(&q, &rows16, 3, &mut c);
        assert!((c[2] - a[2]).abs() < 1e-3);
    }
}
