//! int8 vectors for CPU search: per-vector symmetric scale (max |x| / 127). Dot products in int32:
//! * AVX-512 VNNI (Zen 4/5, Ice Lake+): `vpdpbusd` multiplies u8 x s8, so the query is stored offset by +128 and the
//!   row sum is subtracted: q.x = sum((q+128) x) - 128 sum(x);
//! * AVX2: sign-extend to i16 and `vpmaddwd`;
//! * portable fallback.
use half::f16;
use std::sync::OnceLock;

/// int8 storage that may reuse an f16 allocation (quantised in place, then shrunk with `realloc`).
pub struct I8Buf {
    ptr: *mut i8,
    len: usize,
    layout: std::alloc::Layout,
}
unsafe impl Send for I8Buf {}
unsafe impl Sync for I8Buf {}

impl std::ops::Deref for I8Buf {
    type Target = [i8];
    fn deref(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for I8Buf {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) }
        }
    }
}

impl From<Vec<i8>> for I8Buf {
    fn from(mut v: Vec<i8>) -> Self {
        v.shrink_to_fit();
        let len = v.len();
        let mut v = std::mem::ManuallyDrop::new(v);
        I8Buf { ptr: v.as_mut_ptr(), len, layout: std::alloc::Layout::array::<i8>(len).unwrap() }
    }
}

pub struct QuantRows {
    pub q: I8Buf,        // n × d
    pub scale: Vec<f32>, // per row
    pub rsum: Vec<i32>,  // per row: sum of the int8 values (VNNI offset correction)
}

pub struct QuantQuery {
    pub i8: Vec<i8>,
    pub u8: Vec<u8>, // i8 + 128
    pub scale: f32,
}

fn quantize_into(row: impl Iterator<Item = f32> + Clone, out: &mut [i8]) -> f32 {
    let m = row.clone().fold(0f32, |a, v| a.max(v.abs()));
    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
    for (o, v) in out.iter_mut().zip(row) {
        *o = (v / s).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

pub fn quantize_rows_f16(x: &[f16], d: usize) -> QuantRows {
    use rayon::prelude::*;
    let n = x.len() / d;
    let mut q = vec![0i8; n * d];
    let mut scale = vec![0f32; n];
    let mut rsum = vec![0i32; n];
    q.par_chunks_mut(d).zip(scale.par_iter_mut()).zip(rsum.par_iter_mut()).enumerate().for_each(|(r, ((o, s), rs))| {
        *s = quantize_into(x[r * d..(r + 1) * d].iter().map(|v| v.to_f32()), o);
        *rs = o.iter().map(|&v| v as i32).sum();
    });
    QuantRows { q: q.into(), scale, rsum }
}

/// Quantise an f16 matrix in its own allocation: row r's int8 values go to bytes [r d, (r+1) d), which never overlap
/// rows not yet read (row r' >= r is read from bytes [2 r' d, ...)). Rows run in parallel in waves [2^w, 2^(w+1)):
/// a wave writes only over rows of earlier waves. The buffer is then shrunk to n·d bytes. Peak memory: one f16 copy.
pub fn quantize_rows_f16_in_place(x: Vec<f16>, d: usize) -> QuantRows {
    use rayon::prelude::*;
    let n = x.len() / d;
    let mut x = std::mem::ManuallyDrop::new(x);
    let (ptr, cap) = (x.as_mut_ptr() as usize, x.capacity());
    let mut scale = vec![0f32; n];
    let mut rsum = vec![0i32; n];
    let (sp, rp) = (scale.as_mut_ptr() as usize, rsum.as_mut_ptr() as usize);
    let quant_row = |r: usize| {
        let mut row = vec![0f32; d];
        unsafe {
            let src = std::slice::from_raw_parts((ptr as *const f16).add(r * d), d);
            for (o, v) in row.iter_mut().zip(src) {
                *o = v.to_f32();
            }
            let dst = std::slice::from_raw_parts_mut((ptr as *mut i8).add(r * d), d);
            let s = quantize_into(row.iter().copied(), dst);
            *(sp as *mut f32).add(r) = s;
            *(rp as *mut i32).add(r) = dst.iter().map(|&v| v as i32).sum();
        }
    };
    if n > 0 {
        quant_row(0);
    }
    let mut lo = 1usize;
    while lo < n {
        let hi = (2 * lo).min(n);
        (lo..hi).into_par_iter().for_each(|r| quant_row(r));
        lo = hi;
    }
    let old = std::alloc::Layout::array::<f16>(cap).unwrap();
    let new_size = n * d;
    let p = if new_size > 0 && old.size() > 0 {
        unsafe { std::alloc::realloc(ptr as *mut u8, old, new_size) as *mut i8 }
    } else {
        ptr as *mut i8
    };
    assert!(!p.is_null(), "realloc failed");
    let layout = std::alloc::Layout::from_size_align(new_size, old.align()).unwrap();
    QuantRows { q: I8Buf { ptr: p, len: new_size, layout }, scale, rsum }
}

pub fn quantize_query(q: &[f32]) -> QuantQuery {
    let mut i8v = vec![0i8; q.len()];
    let scale = quantize_into(q.iter().copied(), &mut i8v);
    let u8v = i8v.iter().map(|&v| (v as i16 + 128) as u8).collect();
    QuantQuery { i8: i8v, u8: u8v, scale }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Level {
    Scalar,
    Avx2,
    Vnni512,
}

pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let force = std::env::var("FLASHCLUSTER_SIMD").ok();
            if force.as_deref() == Some("scalar") {
                return Level::Scalar;
            }
            if force.as_deref() != Some("avx2") && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
                return Level::Vnni512;
            }
            if is_x86_feature_detected!("avx2") {
                return Level::Avx2;
            }
        }
        Level::Scalar
    })
}

/// Exact int32 dot q.x of a quantised query with row `r` (contiguous `x`, `rsum` = sum of x).
#[inline]
pub fn dot(q: &QuantQuery, x: &[i8], rsum: i32) -> i32 {
    // single row: the AVX2 widening dot measured faster than VNNI on Zen 5 (no reduction of two zmm per row);
    // VNNI wins when one row load feeds several queries (dot4). FLASHCLUSTER_INT8_DOT=vnni forces VNNI here.
    static VNNI1: OnceLock<bool> = OnceLock::new();
    #[allow(unused_variables)]
    let vnni1 = *VNNI1.get_or_init(|| std::env::var("FLASHCLUSTER_INT8_DOT").as_deref() == Ok("vnni"));
    match level() {
        #[cfg(target_arch = "x86_64")]
        Level::Vnni512 if vnni1 && x.len() % 64 == 0 => unsafe { x86::vnni(q.u8.as_ptr(), x.as_ptr(), x.len()) - 128 * rsum },
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Vnni512 if x.len() % 16 == 0 => unsafe { x86::avx2(q.i8.as_ptr(), x.as_ptr(), x.len()) },
        _ => {
            let _ = rsum;
            q.i8.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
        }
    }
}

/// Four queries against one row (each row byte is loaded once): returns q_i . x for i < 4.
#[inline]
pub fn dot4(q: [&QuantQuery; 4], x: &[i8], rsum: i32) -> [i32; 4] {
    match level() {
        #[cfg(target_arch = "x86_64")]
        Level::Vnni512 if x.len() % 64 == 0 => unsafe {
            let r = x86::vnni4([q[0].u8.as_ptr(), q[1].u8.as_ptr(), q[2].u8.as_ptr(), q[3].u8.as_ptr()], x.as_ptr(), x.len());
            [r[0] - 128 * rsum, r[1] - 128 * rsum, r[2] - 128 * rsum, r[3] - 128 * rsum]
        },
        _ => [dot(q[0], x, rsum), dot(q[1], x, rsum), dot(q[2], x, rsum), dot(q[3], x, rsum)],
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn vnni4(q: [*const u8; 4], x: *const i8, d: usize) -> [i32; 4] {
        use std::arch::x86_64::*;
        let mut acc = [_mm512_setzero_si512(); 4];
        let mut i = 0;
        while i < d {
            let xv = _mm512_loadu_si512(x.add(i) as *const _);
            for j in 0..4 {
                acc[j] = _mm512_dpbusd_epi32(acc[j], _mm512_loadu_si512(q[j].add(i) as *const _), xv);
            }
            i += 64;
        }
        [_mm512_reduce_add_epi32(acc[0]), _mm512_reduce_add_epi32(acc[1]), _mm512_reduce_add_epi32(acc[2]), _mm512_reduce_add_epi32(acc[3])]
    }

    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn vnni(q: *const u8, x: *const i8, d: usize) -> i32 {
        let (mut a, mut b) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        let mut i = 0;
        while i + 128 <= d {
            a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(q.add(i) as *const _), _mm512_loadu_si512(x.add(i) as *const _));
            b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(q.add(i + 64) as *const _), _mm512_loadu_si512(x.add(i + 64) as *const _));
            i += 128;
        }
        if i < d {
            a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(q.add(i) as *const _), _mm512_loadu_si512(x.add(i) as *const _));
        }
        _mm512_reduce_add_epi32(_mm512_add_epi32(a, b))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2(q: *const i8, x: *const i8, d: usize) -> i32 {
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i < d {
            let qa = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.add(i) as *const __m128i));
            let xa = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(i) as *const __m128i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(qa, xa));
            i += 16;
        }
        let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_dot_matches_reference() {
        let d = 768;
        let x: Vec<f16> = (0..d).map(|i| f16::from_f32(((i * 37) % 101) as f32 / 101.0 - 0.5)).collect();
        let qf: Vec<f32> = (0..d).map(|i| ((i * 53) % 97) as f32 / 97.0 - 0.5).collect();
        let r = quantize_rows_f16(&x, d);
        let r2 = quantize_rows_f16_in_place(x.repeat(37), d);
        assert!(r2.q.chunks(d).all(|c| c == &r.q[..]) && r2.scale.iter().all(|&s| s == r.scale[0]));
        let q = quantize_query(&qf);
        let want: i32 = q.i8.iter().zip(r.q.iter()).map(|(&a, &b)| a as i32 * b as i32).sum();
        assert_eq!(dot(&q, &r.q, r.rsum[0]), want);
        let exact: f32 = qf.iter().zip(&x).map(|(a, b)| a * b.to_f32()).sum();
        let approx = want as f32 * q.scale * r.scale[0];
        assert!((exact - approx).abs() < 0.05, "{exact} {approx}");
    }
}
