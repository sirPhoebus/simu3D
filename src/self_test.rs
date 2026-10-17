//! Evidence for the gravity correction, fragment splitting and lifetime, and structural collapse.
//! Prints measured numbers. Exit code is 0 only if every check passes.

use crate::config::Config;
use crate::material::Material;
use crate::math3::{IVec3, Vec3};
use crate::sim::Simulation;
use crate::split;

pub fn run(config: &Config) -> i32 {
    let mut failures = 0;
    failures += split_geometry();
    failures += free_fall(config);
    failures += free_fall_other_gravity();
    failures += blast_split_and_lifetime(config);
    failures += unsupported_structure_falls(config);
    failures += distant_unsupported_falls(config);
    if failures == 0 {
        println!("RESULT=PASS");
        0
    } else {
        println!("RESULT=FAIL failures={failures}");
        1
    }
}

fn split_geometry() -> i32 {
    let side = 10.0 * 1.0e-5_f32.cbrt();
    let size = Vec3::new(side, side, side);
    let pieces = split::pieces(size, 1000, 0.005);
    let parent = size.x * size.y * size.z;
    let sum: f32 = pieces.iter().map(|p| p.volume()).sum();
    let volume_error = (sum - parent).abs();
    let mut max_overlap = 0.0f32;
    let mut not_cube = 0;
    for piece in &pieces {
        if !piece.is_cube() {
            not_cube += 1;
        }
    }
    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            max_overlap = max_overlap.max(split::overlap_volume(pieces[i], pieces[j]));
        }
    }
    println!(
        "split_count={} split_volume_error_m3={volume_error:.3e} split_max_overlap_m3={max_overlap:.3e}",
        pieces.len()
    );
    let mut failed = 0;
    if pieces.len() != 1000 {
        eprintln!("FAIL split count is {}, expected 1000 cubes", pieces.len());
        failed += 1;
    }
    if not_cube != 0 {
        eprintln!("FAIL {not_cube} pieces are not cubes");
        failed += 1;
    }
    if volume_error > 1.0e-6 {
        eprintln!("FAIL split does not conserve volume");
        failed += 1;
    }
    if max_overlap > 1.0e-7 {
        eprintln!("FAIL split pieces overlap");
        failed += 1;
    }
    failed
}

fn free_fall(config: &Config) -> i32 {
    check_free_fall(config.gravity, "gravity_freefall")
}

fn free_fall_other_gravity() -> i32 {
    check_free_fall(2.5, "gravity_freefall_g2_5")
}

fn check_free_fall(gravity: f32, label: &str) -> i32 {
    let mut config = Config::default();
    config.gravity = gravity;
    config.fixed_dt = 1.0 / 120.0;
    let mut sim = Simulation::new(config.clone());
    let y0 = 8.0;
    let probe = sim.spawn_probe(Vec3::new(0.0, y0, 0.0), Vec3::new(0.2, 0.2, 0.2));
    let steps = 120; // 1.000 second
    for _ in 0..steps {
        sim.step(config.fixed_dt);
    }
    let t = steps as f32 * config.fixed_dt;
    let expected_y = y0 - 0.5 * gravity * t * t;
    let expected_v = -gravity * t;
    let pos = sim.body_translation(probe).unwrap();
    let vel = sim.body_linvel(probe).unwrap();
    let y_error = (pos.y - expected_y).abs();
    let v_error = (vel.y - expected_v).abs();
    println!(
        "{label}_y_m={:.6} expected_y_m={expected_y:.6} y_error_m={y_error:.3e} v_error_m_s={v_error:.3e}",
        pos.y
    );
    // 1 mm and 5 mm/s. Float accumulation over 120 steps is far inside that if the correction is right.
    if y_error > 1.0e-3 || v_error > 5.0e-3 {
        eprintln!(
            "FAIL {label}: free-fall does not match y = y0 - 0.5*g*t^2 (y error {y_error:.3e} m, v error {v_error:.3e} m/s)"
        );
        1
    } else {
        0
    }
}

fn blast_split_and_lifetime(config: &Config) -> i32 {
    let mut cfg = config.clone();
    cfg.fragment_lifetime_secs = 5.0;
    cfg.fragment_count = 1000;
    cfg.min_fragment_extent = 0.001;
    cfg.explosion_radius = 0.2;
    cfg.fixed_dt = 1.0 / 120.0;
    let mut sim = Simulation::new(cfg.clone());
    // A single block in the air. Not anchored, so it would fall if we only detached it.
    // The blast splits it before the support pass runs, which is the bazooka path.
    sim.voxels.set(0, 20, 0, Material::Concrete, false);
    let center = sim.voxels.cell_center(IVec3::new(0, 20, 0));
    sim.explode_at(center, 0.12);
    let count = sim.finite_debris_count();
    let grit = sim.grit_count();
    println!("blast_fragment_count={count} grit_kept={grit}");
    let mut failed = 0;
    if count == 0 || count >= 1000 {
        eprintln!("FAIL blast freed {count} cubes, expected a partial crater under 1000");
        failed += 1;
    }
    if grit == 0 {
        eprintln!("FAIL no partial cube remained in place");
        failed += 1;
    }
    if count + grit != 1000 {
        eprintln!("FAIL freed {count} + kept {grit} != 1000");
        failed += 1;
    }
    let speed = sim.mean_debris_speed();
    println!("blast_mean_speed_m_s={speed}");
    if speed < 2.0 {
        eprintln!("FAIL blast pieces are not moving (mean speed {speed} m/s)");
        failed += 1;
    }
    if sim.voxels.blocks.contains_key(&IVec3::new(0, 20, 0)) {
        eprintln!("FAIL blasted cell is still in the standing grid");
        failed += 1;
    }
    let y0 = sim.grit_lowest_y().unwrap_or(0.0);
    let warm = (0.5 / cfg.fixed_dt).floor() as i32;
    for _ in 0..warm {
        sim.step(cfg.fixed_dt);
    }
    let y1 = sim.grit_lowest_y().unwrap_or(y0);
    let drop = y0 - y1;
    println!("grit_drop_0_5s={drop:.3}");
    if drop < 0.4 {
        eprintln!("FAIL kept cubes defied gravity (dropped {drop:.3} m in 0.5 s)");
        failed += 1;
    }
    let steps_before = ((4.9 / cfg.fixed_dt).floor() as i32 - warm).max(0);
    for _ in 0..steps_before {
        sim.step(cfg.fixed_dt);
    }
    let at_4_9 = sim.finite_debris_count();
    println!("fragments_at_4_9s={at_4_9}");
    if at_4_9 != count {
        eprintln!("FAIL fragments despawned before 5 s (count {at_4_9} at 4.9 s)");
        failed += 1;
    }
    let steps_after = ((5.05 - 4.9) / cfg.fixed_dt).ceil() as i32;
    for _ in 0..steps_after {
        sim.step(cfg.fixed_dt);
    }
    let at_5 = sim.finite_debris_count();
    println!("fragments_at_5s={at_5} lifetime_s={}", cfg.fragment_lifetime_secs);
    if at_5 != 0 {
        eprintln!("FAIL fragments still present after 5 s (count {at_5})");
        failed += 1;
    }
    failed
}

fn unsupported_structure_falls(config: &Config) -> i32 {
    let mut cfg = config.clone();
    cfg.fixed_dt = 1.0 / 120.0;
    cfg.gravity = config.gravity;
    let mut sim = Simulation::new(cfg.clone());
    // Column on an anchored base. Erase the base. The three blocks above must become one
    // dynamic structure and fall under the same g. No fragments are left underneath to hold it.
    sim.voxels.set(2, 0, 2, Material::Concrete, true);
    sim.voxels.set(2, 1, 2, Material::Concrete, false);
    sim.voxels.set(2, 2, 2, Material::Concrete, false);
    sim.voxels.set(2, 3, 2, Material::Concrete, false);
    sim.erase_cells(&[IVec3::new(2, 0, 2)]);
    let y0 = match sim.collapsed_centroid_y() {
        Some(y) => y,
        None => {
            eprintln!("FAIL erase of the base did not detach the column");
            return 1;
        }
    };
    let steps = 60; // 0.5 s
    for _ in 0..steps {
        sim.step(cfg.fixed_dt);
    }
    let y1 = sim.collapsed_centroid_y().unwrap_or(y0);
    let t = steps as f32 * cfg.fixed_dt;
    let expected = y0 - 0.5 * cfg.gravity * t * t;
    let error = (y1 - expected).abs();
    println!(
        "collapse_y0_m={y0:.4} collapse_y1_m={y1:.4} expected_y_m={expected:.4} collapse_y_error_m={error:.3e}"
    );
    if error > 2.0e-3 {
        eprintln!("FAIL unsupported column did not follow free-fall (error {error:.3e} m)");
        return 1;
    }
    if sim.collapsed_count() != 1 {
        eprintln!(
            "FAIL expected one falling structure, found {}",
            sim.collapsed_count()
        );
        return 1;
    }
    0
}

fn distant_unsupported_falls(config: &Config) -> i32 {
    let mut cfg = config.clone();
    cfg.fixed_dt = 1.0 / 120.0;
    cfg.adhesive_concrete = 1.0e12;
    cfg.explosion_radius = 0.12;
    let mut sim = Simulation::new(cfg.clone());
    let s = sim.voxels.block_size;
    // A long bridge on one anchored cell. The blast removes only that cell.
    // The far end is outside the local collider rebuild. It must still fall.
    sim.voxels.set(0, 0, 0, Material::Concrete, true);
    for x in 1..16 {
        sim.voxels.set(x, 0, 0, Material::Concrete, false);
    }
    for y in 1..5 {
        sim.voxels.set(15, y, 0, Material::Concrete, false);
    }
    sim.refresh_static();
    let base = sim.voxels.cell_center(IVec3::new(0, 0, 0));
    sim.explode_at(base, cfg.explosion_radius);
    let y0 = match sim.highest_collapsed_y() {
        Some(y) => y,
        None => {
            eprintln!("FAIL blast of the anchor did not detach the far column");
            return 1;
        }
    };
    let steps = 60;
    for _ in 0..steps {
        sim.step(cfg.fixed_dt);
    }
    let y1 = sim.highest_collapsed_y().unwrap_or(y0);
    let drop = y0 - y1;
    println!("distant_collapse_y0_m={y0:.4} distant_collapse_y1_m={y1:.4} drop_m={drop:.4} block_m={s:.4}");
    if drop < 0.5 {
        eprintln!("FAIL far unsupported column stayed in the air (drop {drop:.3} m)");
        return 1;
    }
    0
}

pub fn stress() {
    let rss = || {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    let rest = line.strip_prefix("VmRSS:")?;
                    rest.split_whitespace().next()?.parse::<u64>().ok()
                })
            })
            .unwrap_or(0)
    };
    let mut cfg = Config::default();
    cfg.fragment_lifetime_secs = 5.0;
    cfg.explosion_radius = 2.0;
    cfg.fixed_dt = 1.0 / 120.0;
    println!("rss_start_kb={}", rss());

    // A slab large enough that one blast hits hundreds of blocks.
    let mut sim = Simulation::new(cfg.clone());
    for x in 0..24 {
        for z in 0..24 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
            sim.voxels.set(x, 1, z, Material::Concrete, false);
        }
    }
    sim.refresh_static();
    let (b0, c0, _) = sim.physics_stats();
    println!("before_blast bodies={b0} colliders={c0} rss_kb={}", rss());
    let t0 = std::time::Instant::now();
    sim.explode_at(Vec3::new(3.0, 0.25, 3.0), 2.0);
    let explode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (b1, c1, k1) = sim.physics_stats();
    println!(
        "after_blast explode_ms={explode_ms:.1} bodies={b1} colliders={c1} debris={} collapsed_colliders={k1} rss_kb={}",
        sim.debris_count(),
        rss()
    );
    let steps = (6.0 / cfg.fixed_dt) as i32;
    let t1 = std::time::Instant::now();
    for _ in 0..steps {
        sim.step(cfg.fixed_dt);
    }
    let step_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let (b2, c2, _) = sim.physics_stats();
    println!(
        "after_6s step_ms={step_ms:.1} bodies={b2} colliders={c2} debris={} rss_kb={}",
        sim.debris_count(),
        rss()
    );

    // Second blast on what remains, then wait again. Growth here is a leak.
    sim.explode_at(Vec3::new(3.5, 0.25, 3.5), 2.0);
    for _ in 0..steps {
        sim.step(cfg.fixed_dt);
    }
    let (b3, c3, _) = sim.physics_stats();
    println!(
        "after_second_cycle bodies={b3} colliders={c3} debris={} static_delta_colliders={} rss_kb={}",
        sim.debris_count(),
        c3 as i64 - c0 as i64,
        rss()
    );

    // Unsupported slab: one anchor, a 16x16 plate above it. Destroy the anchor.
    let mut sim = Simulation::new(cfg.clone());
    sim.voxels.set(0, 0, 0, Material::Concrete, true);
    for x in 0..16 {
        for z in 0..16 {
            sim.voxels.set(x, 4, z, Material::Concrete, false);
        }
    }
    // Column so the plate is connected to the anchor.
    for y in 1..4 {
        sim.voxels.set(0, y, 0, Material::Concrete, false);
    }
    sim.erase_cells(&[IVec3::new(0, 0, 0)]);
    let (b, c, k) = sim.physics_stats();
    println!("detached_plate bodies={b} colliders={c} collapsed_colliders={k} rss_kb={}", rss());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::villa;

    #[test]
    fn far_column_falls_after_the_anchor_is_blasted() {
        let cfg = Config::default();
        assert_eq!(distant_unsupported_falls(&cfg), 0);
    }

    #[test]
    fn villa_blast_survives_the_next_steps() {
        let mut cfg = Config::default();
        cfg.fixed_dt = 1.0 / 120.0;
        let mut sim = Simulation::new(cfg.clone());
        villa::build(&mut sim.voxels);
        sim.refresh_static();
        let idle = std::time::Instant::now();
        sim.step(cfg.fixed_dt);
        let (ib, ic, ik) = sim.physics_stats();
        println!(
            "idle_ms={} bodies={ib} colliders={ic} collapsed_colliders={ik}",
            idle.elapsed().as_millis()
        );
        let s = cfg.block_size;
        let impact = Vec3::new((24 + 12) as f32 * s, 1.6, (28 + 40 - 2) as f32 * s);
        println!(
            "impact_xyz={:.2},{:.2},{:.2} blocks={}",
            impact.x,
            impact.y,
            impact.z,
            sim.block_count()
        );
        let t0 = std::time::Instant::now();
        sim.explode_at(impact, cfg.explosion_radius);
        println!(
            "explode_ms={} debris={} blocks={}",
            t0.elapsed().as_millis(),
            sim.debris_count(),
            sim.block_count()
        );
        let horiz = sim.mean_cluster_horizontal_speed();
        println!("cluster_horizontal_m_s={horiz:.2}");
        assert!(
            horiz > 4.0,
            "blast chunks have no sideways speed ({horiz:.2} m/s); the impact force is missing"
        );
        sim.take_static_meshes();
        let ti = std::time::Instant::now();
        let drawn = sim.instances().len();
        println!(
            "instances={drawn} instance_build_us={}",
            ti.elapsed().as_micros()
        );
        assert!(drawn < 200_000, "shell surface culling left {drawn} instances");
        let tm = std::time::Instant::now();
        let opaque = crate::mesh::build(&sim.voxels, false);
        let glass = crate::mesh::build(&sim.voxels, true);
        println!(
            "mesh_us={} opaque_tris={} glass_tris={}",
            tm.elapsed().as_micros(),
            opaque.indices.len() / 3,
            glass.indices.len() / 3
        );
        for i in 0..20 {
            let t = std::time::Instant::now();
            sim.step(cfg.fixed_dt);
            let ms = t.elapsed().as_millis();
            if i < 8 || i == 19 {
                let (b, c, k) = sim.physics_stats();
                println!(
                    "step={i} us={} debris={} bodies={b} colliders={c} collapsed_colliders={k}",
                    t.elapsed().as_micros(),
                    sim.debris_count()
                );
            }
            if i > 0 && !cfg!(debug_assertions) {
                assert!(
                    ms < 8,
                    "step {i} took {ms} ms; a 120 Hz step must stay under 8 ms to hold 60 FPS"
                );
            }
        }
        let blocks_after = sim.block_count();
        for _ in 0..120 {
            sim.step(cfg.fixed_dt);
        }
        assert!(
            sim.block_count() <= blocks_after,
            "voxels were written back into the hole: {} then {}",
            blocks_after,
            sim.block_count()
        );
    }
}
