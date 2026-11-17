//! `gpu-fit`: multi-seed e-means on the GPU, a canonical (medoid) seed, nested coarse levels and per-node stability.
//!
//! Input: an n × d '<f2' .npy of L2-normalised rows. For each seed: e-means (quality law) with the chosen
//! assignment precision, then an exact f32 assignment of every row. The canonical seed is the one with the highest
//! mean ARI to the other seeds (or, with --canonical objective, the lowest objective); --spherical unit-normalises
//! the leaf centroids before the final assignment. Coarse levels (--level-method): `flat` (default) fits e-means at
//! each coarse k on all rows; the first coarse level keeps its own row labels and coarser levels nest under it by
//! majority (leaves are linked to it by majority, not nested); `nested` nests every level under the leaves by
//! majority; `weighted` runs k-means over leaf centroids. Stability per node compares the canonical run with the
//! other seeds at every level. Measured on 2.43M text chunks (k = 16384 / 1024 / 64, 5 seeds), share of rows in
//! stable nodes at 1024 / 64: weighted 13% / 11%, flat nested under the leaves 31% / 33%, flat with 64 nested under
//! 1024 69% / 70%, each flat fit alone 72% / 88%.
//!
//! Outputs in --out:
//!   seed_<s>/centroids.f32.npy, seed_<s>/assign.u32.npy        every seed's leaves
//!   centroids.f32.npy        k × d canonical leaf centroids
//!   assign.u32.npy           n leaf id per row (canonical)
//!   centroids_p<P>.f32.npy   k × P prefix centroids of the same partition (--prefix-dims)
//!   stability.f32.npy        n: stability of the row's canonical leaf
//!   level_<i>_centroids.f32.npy, level_<i>_parent.u32.npy, level_<i>_assign.u32.npy   coarse level i (finest
//!                            first): centroids, parent of each node of the next finer level (leaf → level 0 →
//!                            level 1 ...; with --level-method flat the leaf → level 0 link is by majority only)
//!                            and every row's node at this level
//!   node_stability_<name>.f32.npy  per node, for "leaf" and every coarse level
//!   fit.json
use crate::gpu::{device_mem_used_gb, init, Gpu, GpuEngine};
use crate::hier::{self, build_levels, node_stability};
use crate::io;
use crate::kmeans::{adjusted_rand, balance, emeans_with, BatchLaw};
use anyhow::{bail, Result};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

#[derive(clap::Args)]
pub struct GpuFitArgs {
    /// n × d '<f2' .npy, rows L2-normalised
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub k: usize,
    #[arg(long, default_value_t = 5)]
    pub seeds: u64,
    #[arg(long, default_value_t = 1234)]
    pub seed0: u64,
    #[arg(long, default_value_t = 5.0)]
    pub nepochs: f64,
    /// e-means assignment precision: f32, tf32, f16, bf16, int8 (updates and the final assignment are f32)
    #[arg(long, default_value = "f16")]
    pub precision: String,
    /// coarse levels over the leaves, finest first, e.g. "1024,64" (empty = none)
    #[arg(long, default_value = "")]
    pub levels: String,
    /// how coarse levels are built: "flat" (e-means at each k on all rows, then each finer node joins the coarser
    /// node holding most of its rows) or "weighted" (weighted k-means over the finer level's centroids)
    #[arg(long, default_value = "flat")]
    pub level_method: String,
    /// also write prefix centroids in the first P dimensions (0 = off)
    #[arg(long, default_value_t = 0)]
    pub prefix_dims: usize,
    /// unit-normalise the leaf centroids before the final assignment (spherical: nearest centroid = largest inner
    /// product, which is how a router scores queries against unit centroids)
    #[arg(long)]
    pub spherical: bool,
    /// canonical seed: "medoid" (highest mean ARI to the other seeds) or "objective" (lowest objective)
    #[arg(long, default_value = "medoid")]
    pub canonical: String,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub out: PathBuf,
}

fn write_f32(p: &PathBuf, shape: &[usize], v: &[f32]) -> Result<()> {
    io::write_npy_bytes(p, "<f4", shape, io::as_bytes(v))
}
fn write_u32(p: &PathBuf, v: &[u32]) -> Result<()> {
    io::write_npy_bytes(p, "<u4", &[v.len()], io::as_bytes(v))
}

pub fn gpu_fit(a: GpuFitArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let levels: Vec<usize> = a.levels.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()?;
    if a.seeds < 1 {
        bail!("--seeds must be >= 1");
    }
    let name = init(a.device, false)?;
    let t0 = Instant::now();
    let (shape, data) = io::read_npy_f16(&a.data)?;
    if shape.len() != 2 {
        bail!("expected a 2-d array, got {shape:?}");
    }
    let (n, d, k) = (shape[0], shape[1], a.k);
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("loaded {n} x {d} in {load_s:.1}s on {name}");
    let t0 = Instant::now();
    let g = Gpu::upload(&data, n, d)?;
    let upload_s = t0.elapsed().as_secs_f64();

    struct Run { seed: u64, c: Vec<f32>, lab: Vec<u32>, levels: Vec<hier::Level> }
    let mut runs: Vec<Run> = vec![];
    let mut seed_rows = vec![];
    for s in 0..a.seeds {
        let seed = a.seed0 + s;
        let t = Instant::now();
        let (mut c, recipe) = emeans_with(&mut GpuEngine::with_precision(&g, &a.precision)?, k, a.nepochs, BatchLaw::Quality, seed);
        if a.spherical {
            c.chunks_mut(d).for_each(|r| {
                let nrm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
                if nrm > 0.0 {
                    r.iter_mut().for_each(|v| *v /= nrm);
                }
            });
        }
        let fit_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (obj, lab) = g.assign(&c, k)?;
        let assign_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut lv: Vec<hier::Level> = vec![];
        let mut purity = vec![];
        match a.level_method.as_str() {
            "weighted" => lv = build_levels(&c, &hier::sizes(&lab, k), d, &levels, seed),
            // flat: each coarse level is its own e-means fit on all rows. The first coarse level keeps its flat row
            // labels (the leaves link to it by majority, for reference); every coarser level nests under the one
            // below it by majority, so the coarse tree is exact and leaves are only linked.
            // nested: the same fits, but every level, the first included, nests under the leaves by majority.
            "flat" | "nested" => {
                let (mut child_lab, mut child_c, mut child_k) = (lab.clone(), c.clone(), k);
                for (li, &kl) in levels.iter().enumerate() {
                    let (cl, _) = emeans_with(&mut GpuEngine::with_precision(&g, &a.precision)?, kl, a.nepochs, BatchLaw::Quality, seed + 7919 * (li as u64 + 1));
                    let (obj_l, lab_l) = g.assign(&cl, kl)?;
                    let (parent_of, agree) = hier::nest_majority(&child_lab, child_k, &lab_l);
                    purity.push(json!({"k": kl, "flat_objective": obj_l, "rows_agreeing_with_flat_fit": agree}));
                    if a.level_method == "flat" && li == 0 {
                        child_lab = lab_l.clone();
                        child_c = cl.clone();
                        lv.push(hier::Level { k: kl, centroids: cl, parent_of, objective: obj_l, assign: lab_l });
                    } else {
                        let nested = hier::nested_centroids(&child_c, &hier::sizes(&child_lab, child_k), &parent_of, kl, d);
                        child_lab = child_lab.iter().map(|&x| parent_of[x as usize]).collect();
                        child_c = nested.clone();
                        lv.push(hier::Level { k: kl, centroids: nested, parent_of, objective: obj_l, assign: child_lab.clone() });
                    }
                    child_k = kl;
                }
            }
            m => bail!("unknown --level-method {m}"),
        }
        let levels_s = t.elapsed().as_secs_f64();
        let dir = a.out.join(format!("seed_{seed}"));
        std::fs::create_dir_all(&dir)?;
        write_f32(&dir.join("centroids.f32.npy"), &[k, d], &c)?;
        write_u32(&dir.join("assign.u32.npy"), &lab)?;
        let row = json!({"seed": seed, "fit_seconds": fit_s, "assign_seconds": assign_s, "levels_seconds": levels_s,
            "objective": obj, "balance": balance(&lab, k), "recipe": recipe,
            "level_objectives": lv.iter().map(|l| l.objective).collect::<Vec<_>>(), "level_nesting": purity});
        eprintln!("{row}");
        seed_rows.push(row);
        runs.push(Run { seed, c, lab, levels: lv });
    }
    drop(g);
    let device_mem = device_mem_used_gb();

    // canonical seed = highest mean ARI to the others (leaf level)
    let m = runs.len();
    let mut ari = vec![vec![1.0f64; m]; m];
    for i in 0..m {
        for j in i + 1..m {
            let v = adjusted_rand(&runs[i].lab, &runs[j].lab);
            ari[i][j] = v;
            ari[j][i] = v;
        }
    }
    let mean_ari: Vec<f64> = (0..m).map(|i| if m == 1 { 1.0 } else { (0..m).filter(|&j| j != i).map(|j| ari[i][j]).sum::<f64>() / (m - 1) as f64 }).collect();
    let objs: Vec<f64> = seed_rows.iter().map(|r| r["objective"].as_f64().unwrap()).collect();
    let ci = match a.canonical.as_str() {
        "medoid" => (0..m).max_by(|&x, &y| mean_ari[x].partial_cmp(&mean_ari[y]).unwrap().then(y.cmp(&x))).unwrap(),
        "objective" => (0..m).min_by(|&x, &y| objs[x].partial_cmp(&objs[y]).unwrap().then(x.cmp(&y))).unwrap(),
        c => bail!("unknown --canonical {c}"),
    };
    let canon = &runs[ci];

    // per-row labels at every level for every run: leaf, then coarse levels finest first
    let per_level = |r: &Run| -> Vec<Vec<u32>> {
        let mut out = vec![r.lab.clone()];
        for l in &r.levels {
            let prev = out.last().unwrap();
            let next = if l.assign.is_empty() { prev.iter().map(|&x| l.parent_of[x as usize]).collect() } else { l.assign.clone() };
            out.push(next);
        }
        out
    };
    let canon_lv = per_level(canon);
    let others_lv: Vec<Vec<Vec<u32>>> = runs.iter().enumerate().filter(|(i, _)| *i != ci).map(|(_, r)| per_level(r)).collect();
    let names: Vec<String> = std::iter::once("leaf".to_string()).chain((0..levels.len()).map(|i| format!("level_{i}"))).collect();
    let mut stab_rows = vec![];
    let mut leaf_stab = vec![];
    for (li, nm) in names.iter().enumerate() {
        let others: Vec<&[u32]> = others_lv.iter().map(|o| o[li].as_slice()).collect();
        let st = node_stability(&canon_lv[li], &others);
        let r = &canon_lv[li];
        let share = r.iter().filter(|&&x| st[x as usize] >= 0.5).count() as f64 / n as f64;
        let mut s = st.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let aris: Vec<f64> = others.iter().map(|o| adjusted_rand(r, o)).collect();
        stab_rows.push(json!({"level": nm, "nodes": st.len(), "node_stability_median": s[s.len() / 2],
            "share_rows_in_stable_nodes(>=0.5)": share,
            "ari_vs_canonical_mean": if aris.is_empty() { None } else { Some(aris.iter().sum::<f64>() / aris.len() as f64) }}));
        write_f32(&a.out.join(format!("node_stability_{nm}.f32.npy")), &[st.len()], &st.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
        if li == 0 {
            leaf_stab = st;
        }
    }

    write_f32(&a.out.join("centroids.f32.npy"), &[k, d], &canon.c)?;
    write_u32(&a.out.join("assign.u32.npy"), &canon.lab)?;
    let row_stab: Vec<f32> = canon.lab.iter().map(|&l| leaf_stab[l as usize] as f32).collect();
    write_f32(&a.out.join("stability.f32.npy"), &[n], &row_stab)?;
    for (i, l) in canon.levels.iter().enumerate() {
        write_f32(&a.out.join(format!("level_{i}_centroids.f32.npy")), &[l.k, d], &l.centroids)?;
        write_u32(&a.out.join(format!("level_{i}_parent.u32.npy")), &l.parent_of)?;
        write_u32(&a.out.join(format!("level_{i}_assign.u32.npy")), &canon_lv[i + 1])?;
    }
    if a.prefix_dims > 0 {
        let pc = hier::prefix_centroids(&data, d, &canon.lab, k, a.prefix_dims);
        write_f32(&a.out.join(format!("centroids_p{}.f32.npy", a.prefix_dims)), &[k, a.prefix_dims], &pc)?;
    }
    let rep = json!({"engine": "flashcluster gpu-fit (e-means, quality law)", "device": name, "data": a.data, "n": n, "d": d, "k": k,
        "nepochs": a.nepochs, "assign_precision": a.precision, "final_assignment": "exact f32", "levels": levels, "level_method": a.level_method,
        "prefix_dims": a.prefix_dims, "load_seconds": load_s, "upload_seconds": upload_s, "device_mem_gb": device_mem,
        "peak_rss_gb": io::peak_rss_gb(), "seeds": seed_rows, "ari_matrix": ari, "mean_ari": mean_ari,
        "canonical_seed": canon.seed, "canonical_rule": a.canonical, "spherical": a.spherical, "stability": stab_rows, "version": env!("CARGO_PKG_VERSION")});
    io::write_json(&a.out.join("fit.json"), &rep)?;
    println!("{}", json!({"canonical_seed": canon.seed, "canonical_rule": a.canonical, "spherical": a.spherical, "mean_ari": mean_ari, "stability": rep["stability"]}));
    Ok(())
}
