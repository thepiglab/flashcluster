//! Matrices and .npy I/O (little-endian f16 / f32 / f64 / u32, C order), JSON helpers, peak RSS.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Minimal .npy reader for little-endian float arrays (f2, f4, f8), C order. Returns (shape, f32 data).
pub fn read_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_npy(&b).with_context(|| format!("parse {}", path.display()))
}

fn header_field<'a>(h: &'a str, key: &str) -> Result<&'a str> {
    let at = h.find(&format!("'{key}'")).with_context(|| format!("npy header lacks {key}"))?;
    Ok(h[at + key.len() + 2..].trim_start_matches([':', ' ']))
}

pub fn parse_npy(b: &[u8]) -> Result<(Vec<usize>, Vec<f32>)> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let (hlen, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let h = std::str::from_utf8(&b[off..off + hlen])?;
    let descr = header_field(h, "descr")?;
    let descr = descr.trim_start_matches('\'').split('\'').next().unwrap_or("");
    if header_field(h, "fortran_order")?.starts_with("True") {
        bail!("fortran_order arrays are not supported");
    }
    let s = header_field(h, "shape")?;
    let s = &s[s.find('(').context("shape")? + 1..s.find(')').context("shape")?];
    let shape: Vec<usize> = s.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
    let n: usize = shape.iter().product();
    let body = &b[off + hlen..];
    let data: Vec<f32> = match descr {
        "<f2" => body[..n * 2].chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "<f4" => body[..n * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "<f8" => body[..n * 8].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        d => bail!("unsupported dtype {d}"),
    };
    Ok((shape, data))
}

/// Row-major (n, d) float32 matrix.
#[derive(Clone)]
pub struct Mat {
    pub data: Vec<f32>,
    pub n: usize,
    pub d: usize,
}

impl Mat {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
    pub fn gather(&self, idx: &[u32]) -> Mat {
        let mut data = Vec::with_capacity(idx.len() * self.d);
        for &i in idx {
            data.extend_from_slice(self.row(i as usize));
        }
        Mat { data, n: idx.len(), d: self.d }
    }
    pub fn normalise(&mut self) {
        crate::linalg::normalise_rows(&mut self.data, self.d);
    }
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&std::fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

pub fn write_json(path: &Path, v: &serde_json::Value) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}

pub fn peak_rss_gb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let kb = ru.ru_maxrss as f64;
    if cfg!(target_os = "macos") { kb / 2f64.powi(30) } else { kb / 2f64.powi(20) }
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let sh = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {sh}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

/// Write a C-order .npy from raw little-endian element bytes (descr e.g. "<f2", "<f4", "<u4").
pub fn write_npy_bytes(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&npy_header(descr, shape))?;
    f.write_all(data)?;
    Ok(())
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Read a little-endian 2-byte-element .npy ('<f2') straight into an f16 buffer (no intermediate copy).
pub fn read_npy_f16(path: &Path) -> Result<(Vec<usize>, Vec<half::f16>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut pre = [0u8; 12];
    f.read_exact(&mut pre[..10])?;
    if &pre[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let hlen = if pre[6] == 1 {
        u16::from_le_bytes([pre[8], pre[9]]) as usize
    } else {
        f.read_exact(&mut pre[10..12])?;
        u32::from_le_bytes([pre[8], pre[9], pre[10], pre[11]]) as usize
    };
    let mut h = vec![0u8; hlen];
    f.read_exact(&mut h)?;
    let h = String::from_utf8(h)?;
    if !h.contains("'<f2'") {
        bail!("{}: expected <f2, header {h}", path.display());
    }
    let s = header_field(&h, "shape")?;
    let s = &s[s.find('(').context("shape")? + 1..s.find(')').context("shape")?];
    let shape: Vec<usize> = s.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
    let n: usize = shape.iter().product();
    let mut v = vec![half::f16::ZERO; n];
    let buf = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 2) };
    f.read_exact(buf)?;
    Ok((shape, v))
}

/// Read a '<u4' .npy.
pub fn read_npy_u32(path: &Path) -> Result<(Vec<usize>, Vec<u32>)> {
    let b = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let h = std::str::from_utf8(&b[10..10 + hlen])?;
    if !h.contains("'<u4'") {
        bail!("{}: expected <u4", path.display());
    }
    let s = header_field(h, "shape")?;
    let s = &s[s.find('(').context("shape")? + 1..s.find(')').context("shape")?];
    let shape: Vec<usize> = s.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
    let n: usize = shape.iter().product();
    Ok((shape, b[10 + hlen..10 + hlen + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
}
