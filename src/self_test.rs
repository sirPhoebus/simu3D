//! Evidence for the gravity correction, fragment splitting and lifetime, and structural collapse.
//! Prints measured numbers. Exit code is 0 only if every check passes.

use crate::config::Config;
use crate::material::Material;
use crate::math3::{IVec3, Vec3};
use crate::sim::{FrameInput, Simulation};
use crate::split;

/// A release villa step has to stay under this to hold 60 FPS at 120 Hz physics.
const VILLA_STEP_BUDGET_MS: u128 = 8;

pub fn run(config: &Config) -> i32 {
    let mut failures = 0;
    failures += split_geometry();
    failures += free_fall(config);
    failures += free_fall_other_gravity();
    failures += blast_split_and_lifetime(config);
    failures += unsupported_structure_falls(config);
    failures += distant_unsupported_falls(config);
    failures += resting_stack_stays_within_a_millimetre(config);
    failures += side_contact_is_not_a_floor(config);
    failures += untouched_chunk_keeps_its_colliders(config);
    failures += glass_bond_breaks_and_metal_bond_holds(config);
    failures += player_falls_through_an_erased_floor(config);
    failures += player_stands_on_settled_rubble(config);
    failures += player_is_stopped_by_a_wall(config);
    failures += blast_colliders_match_voxels_after_lifetime(config);
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
    if count < 8 || count > 350 {
        eprintln!("FAIL blast freed {count} pieces; a hit block must explode into at most 350 smaller blocks");
        failed += 1;
    }
    if grit != 0 {
        eprintln!("FAIL {grit} cubes stayed fused after the block was hit");
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
    let warm = (0.5 / cfg.fixed_dt).floor() as i32;
    for _ in 0..warm {
        sim.step(cfg.fixed_dt);
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
    println!(
        "fragments_at_5s={at_5} lifetime_s={}",
        cfg.fragment_lifetime_secs
    );
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

fn fixed_step(config: &Config) -> Config {
    let mut cfg = config.clone();
    cfg.fixed_dt = 1.0 / 120.0;
    cfg
}

fn step_for(sim: &mut Simulation, seconds: f32) {
    let steps = (seconds / sim.config.fixed_dt).round().max(0.0) as i32;
    for _ in 0..steps {
        sim.step(sim.config.fixed_dt);
    }
}

fn feet_y(sim: &Simulation) -> f32 {
    sim.feet_y()
}

fn wait_until_grounded(sim: &mut Simulation, seconds: f32) -> bool {
    let steps = (seconds / sim.config.fixed_dt).round().max(1.0) as i32;
    for _ in 0..steps {
        sim.step(sim.config.fixed_dt);
        if sim.player.grounded {
            return true;
        }
    }
    sim.player.grounded
}

/// Two cubes on a static floor, one stacked on the other, stay within a millimetre for a second.
fn resting_stack_stays_within_a_millimetre(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.2;
    let mut sim = Simulation::new(cfg.clone());
    let s = sim.voxels.block_size;
    for x in -1..=1 {
        for z in -1..=1 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
        }
    }
    sim.refresh_static();
    let half = Vec3::new(s, s, s) * 0.5;
    let lower = sim.spawn_probe(Vec3::new(0.0, 1.5 * s, 0.0), half);
    let upper = sim.spawn_probe(Vec3::new(0.0, 2.5 * s, 0.0), half);
    step_for(&mut sim, 0.5);
    let lower_y0 = sim.body_translation(lower).map(|p| p.y);
    let upper_y0 = sim.body_translation(upper).map(|p| p.y);
    let (Some(lower_y0), Some(upper_y0)) = (lower_y0, upper_y0) else {
        eprintln!("FAIL resting stack lost a cube during settle");
        return 1;
    };
    println!("rest_lower_y_m={lower_y0:.6} rest_upper_y_m={upper_y0:.6}");
    if lower_y0 <= s {
        eprintln!("FAIL lower cube fell through the floor during settle (y {lower_y0:.4} m)");
        return 1;
    }
    if upper_y0 < lower_y0 + s * 0.4 {
        eprintln!(
            "FAIL upper cube is not stacked on the lower one (y {upper_y0:.4} vs {lower_y0:.4})"
        );
        return 1;
    }
    step_for(&mut sim, 1.0);
    let lower_y1 = sim.body_translation(lower).map(|p| p.y).unwrap_or(lower_y0);
    let upper_y1 = sim.body_translation(upper).map(|p| p.y).unwrap_or(upper_y0);
    let lower_drift = (lower_y1 - lower_y0).abs();
    let upper_drift = (upper_y1 - upper_y0).abs();
    println!("rest_lower_drift_m={lower_drift:.3e} rest_upper_drift_m={upper_drift:.3e}");
    if lower_drift > 1.0e-3 || upper_drift > 1.0e-3 {
        eprintln!(
            "FAIL stacked cubes did not stay at rest (lower drift {lower_drift:.3e} m, upper drift {upper_drift:.3e} m)"
        );
        return 1;
    }
    if upper_y1 < lower_y1 + s * 0.4 {
        eprintln!("FAIL stack collapsed during the measured second");
        return 1;
    }
    0
}

/// A cube touching a wall and nothing else falls the free-fall distance.
fn side_contact_is_not_a_floor(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.2;
    let mut sim = Simulation::new(cfg.clone());
    let s = sim.voxels.block_size;
    for y in 0..40 {
        sim.voxels.set(0, y, 0, Material::Concrete, true);
    }
    sim.refresh_static();
    let half = s * 0.5;
    let y0 = 3.0;
    // The wall cell occupies x in [0, s]. Overlap its face by 1 mm. No voxel is under the cube.
    let probe = sim.spawn_probe(
        Vec3::new(s + half - 0.001, y0, s * 0.5),
        Vec3::new(half, half, half),
    );
    let steps = 60;
    step_for(&mut sim, steps as f32 * cfg.fixed_dt);
    let t = steps as f32 * cfg.fixed_dt;
    let expected = y0 - 0.5 * cfg.gravity * t * t;
    let y1 = sim.body_translation(probe).map(|p| p.y).unwrap_or(y0);
    let error = (y1 - expected).abs();
    println!("side_touch_y_m={y1:.4} side_touch_expected_y_m={expected:.4} side_touch_error_m={error:.3e}");
    if error > 2.0e-3 {
        eprintln!("FAIL a sideways touch held the cube up (error {error:.3e} m)");
        return 1;
    }
    0
}

/// A blast in one chunk leaves another chunk's collider handles, and the heightfield, alone.
fn untouched_chunk_keeps_its_colliders(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.2;
    cfg.adhesive_concrete = 1.0e12;
    cfg.blast_impulse = 2000.0;
    let mut sim = Simulation::new(cfg.clone());
    sim.voxels.set(0, 0, 0, Material::Concrete, true);
    sim.voxels.set(0, 1, 0, Material::Concrete, true);
    sim.voxels.set(40, 0, 0, Material::Concrete, true);
    sim.voxels.set(40, 1, 0, Material::Concrete, true);
    sim.refresh_static();
    let far_cell = IVec3::new(40, 0, 0);
    let near_cell = IVec3::new(0, 0, 0);
    let kept_cell = IVec3::new(0, 1, 0);
    let far_before = sim.static_chunk_handles(far_cell);
    let terrain_before = sim.terrain_collider_handle();
    if far_before.is_empty() {
        eprintln!("FAIL far chunk had no collider before the blast");
        return 1;
    }
    let at = sim.voxels.cell_center(near_cell);
    sim.explode_at(at, 0.09);
    if sim.voxels.blocks.contains_key(&near_cell) {
        eprintln!("FAIL blasted cell is still in the grid");
        return 1;
    }
    if !sim.voxels.blocks.contains_key(&kept_cell) || !sim.voxels.blocks.contains_key(&far_cell) {
        eprintln!("FAIL the blast removed a cell outside its chunk");
        return 1;
    }
    if sim.static_chunk_handles(far_cell) != far_before {
        eprintln!("FAIL untouched chunk collider handles changed");
        return 1;
    }
    if sim.terrain_collider_handle() != terrain_before {
        eprintln!("FAIL blast rebuilt the terrain heightfield");
        return 1;
    }
    sim.step(cfg.fixed_dt);
    if sim.fixed_collider_covers_cell(near_cell) {
        eprintln!("FAIL removed cell is still covered by a fixed collider");
        return 1;
    }
    if !sim.fixed_collider_covers_cell(kept_cell) {
        eprintln!("FAIL the cell that stayed in the blasted chunk lost its collider");
        return 1;
    }
    if !sim.static_colliders_match_voxels() {
        eprintln!("FAIL static colliders do not match the voxels after a local rebuild");
        return 1;
    }
    println!(
        "untouched_chunk_handles={} terrain_rebuilt=0",
        far_before.len()
    );
    0
}

/// The same hit separates glass (adhesion 0) and leaves metal (adhesion 260) bonded.
fn glass_bond_breaks_and_metal_bond_holds(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.2;
    cfg.blast_impulse = 2000.0;
    cfg.adhesive_glass = 0.0;
    cfg.adhesive_metal = 260.0;
    cfg.fragment_count = 8;
    let mut sim = Simulation::new(cfg);
    let glass_hit = IVec3::new(0, 2, 0);
    let glass_next = IVec3::new(1, 2, 0);
    let metal_hit = IVec3::new(0, 2, 30);
    let metal_next = IVec3::new(1, 2, 30);
    sim.voxels
        .set(glass_hit.x, glass_hit.y, glass_hit.z, Material::Glass, true);
    sim.voxels.set(
        glass_next.x,
        glass_next.y,
        glass_next.z,
        Material::Glass,
        true,
    );
    sim.voxels
        .set(metal_hit.x, metal_hit.y, metal_hit.z, Material::Metal, true);
    sim.voxels.set(
        metal_next.x,
        metal_next.y,
        metal_next.z,
        Material::Metal,
        true,
    );
    sim.refresh_static();
    // Radius stays inside the hit cell. The neighbour is a bond, not a second direct hit.
    sim.explode_at(sim.voxels.cell_center(glass_hit), 0.09);
    sim.explode_at(sim.voxels.cell_center(metal_hit), 0.09);
    let mut failed = 0;
    if sim.voxels.blocks.contains_key(&glass_hit) || sim.voxels.blocks.contains_key(&metal_hit) {
        eprintln!("FAIL a directly hit cell stayed in the grid");
        failed += 1;
    }
    if sim.voxels.blocks.contains_key(&glass_next) {
        eprintln!("FAIL glass neighbour stayed bonded at adhesion 0");
        failed += 1;
    }
    if !sim.voxels.blocks.contains_key(&metal_next) {
        eprintln!("FAIL metal neighbour separated at adhesion 260");
        failed += 1;
    }
    println!(
        "bond_glass_neighbour_gone={} bond_metal_neighbour_stays={}",
        !sim.voxels.blocks.contains_key(&glass_next) as u8,
        sim.voxels.blocks.contains_key(&metal_next) as u8
    );
    failed
}

fn player_falls_through_an_erased_floor(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.5;
    let mut sim = Simulation::new(cfg.clone());
    let mut cells = Vec::new();
    for x in -3..=3 {
        for z in -3..=3 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
            cells.push(IVec3::new(x, 0, z));
        }
    }
    sim.refresh_static();
    let top = sim.voxels.block_size;
    sim.place_player(Vec3::new(0.0, top, 0.0), 0.0, 0.0);
    if !wait_until_grounded(&mut sim, 1.0) {
        eprintln!("FAIL player never landed on the floor");
        return 1;
    }
    let y0 = feet_y(&sim);
    sim.erase_cells(&cells);
    step_for(&mut sim, 0.5);
    let drop = y0 - feet_y(&sim);
    println!("player_floor_drop_m={drop:.4}");
    if drop <= 0.4 {
        eprintln!("FAIL player did not fall through the erased floor (drop {drop:.3} m)");
        return 1;
    }
    0
}

fn player_stands_on_settled_rubble(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.5;
    let mut sim = Simulation::new(cfg.clone());
    for x in -3..=3 {
        for z in -3..=3 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
        }
    }
    sim.voxels.set(0, 1, 0, Material::Concrete, true);
    for x in -1..=1 {
        for z in -1..=1 {
            sim.voxels.set(x, 2, z, Material::Concrete, false);
        }
    }
    sim.refresh_static();
    let floor_top = sim.voxels.block_size;
    let block = sim.voxels.block_size;
    sim.erase_cells(&[IVec3::new(0, 1, 0)]);
    if sim.collapsed_count() != 1 {
        eprintln!(
            "FAIL erasing the anchor detached {} structures, expected 1",
            sim.collapsed_count()
        );
        return 1;
    }
    // The first steps are still a free fall: |vy| starts near zero. Wait until the
    // piece is down on the floor and no longer moving fast.
    let rest_top = floor_top + block;
    let mut settled = false;
    for _ in 0..(3.0 / cfg.fixed_dt).round() as i32 {
        sim.step(cfg.fixed_dt);
        let Some(top) = sim.collapsed_top_y() else {
            break;
        };
        let vy = sim.collapsed_vertical_speed().unwrap_or(1.0);
        // The first bounce still clears the floor. Accept the piece only once it is back down.
        if top <= rest_top + 0.05 && vy.abs() < 0.25 {
            settled = true;
            break;
        }
    }
    let Some(top) = sim.collapsed_top_y() else {
        eprintln!("FAIL collapsed piece has no top");
        return 1;
    };
    let Some(origin) = sim.collapsed_origin() else {
        eprintln!("FAIL collapsed piece has no origin");
        return 1;
    };
    println!(
        "rubble_top_y_m={top:.4} rubble_vy_m_s={:.4} settled={settled}",
        sim.collapsed_vertical_speed().unwrap_or(f32::NAN)
    );
    if top < floor_top + sim.voxels.block_size * 0.5 {
        eprintln!(
            "FAIL rubble is not standing above the floor (top {top:.3} m, floor {floor_top:.3} m)"
        );
        return 1;
    }
    sim.place_player(Vec3::new(origin.x, top + 0.02, origin.z), 0.0, 0.0);
    step_for(&mut sim, 0.5);
    let Some(top_after) = sim.collapsed_top_y() else {
        eprintln!("FAIL rubble disappeared under the player");
        return 1;
    };
    let feet = feet_y(&sim);
    let gap = (feet - top_after).abs();
    println!(
        "player_on_rubble_feet_m={feet:.4} rubble_top_m={top_after:.4} gap_m={gap:.4} grounded={}",
        sim.player.grounded
    );
    if gap > 0.25 {
        eprintln!("FAIL player is not standing on the rubble (gap {gap:.3} m)");
        return 1;
    }
    if !sim.player.grounded {
        eprintln!("FAIL player is not grounded on the rubble");
        return 1;
    }
    0
}

fn player_is_stopped_by_a_wall(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.5;
    cfg.walk_speed = 4.3;
    let mut sim = Simulation::new(cfg.clone());
    for x in -2..=2 {
        for z in 1..=8 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
        }
    }
    for x in -4..=4 {
        for y in 0..=12 {
            sim.voxels.set(x, y, 0, Material::Stone, true);
        }
    }
    sim.refresh_static();
    let top = sim.voxels.block_size;
    let start_z = 3.0;
    sim.place_player(Vec3::new(0.0, top, start_z), 0.0, 0.0);
    if !wait_until_grounded(&mut sim, 1.0) {
        eprintln!("FAIL player never landed before the wall");
        return 1;
    }
    let z_before = sim.player.pos.z;
    let y_before = feet_y(&sim);
    sim.set_input(FrameInput {
        forward: 1.0,
        ..FrameInput::default()
    });
    step_for(&mut sim, 1.0);
    let z_after = sim.player.pos.z;
    let moved = z_before - z_after;
    println!(
        "player_wall_z0={z_before:.3} player_wall_z1={z_after:.3} player_wall_moved_m={moved:.3}"
    );
    if moved <= 0.3 {
        eprintln!("FAIL player did not walk toward the wall (moved {moved:.3} m)");
        return 1;
    }
    if z_after <= 0.7 {
        eprintln!("FAIL wall did not stop the player (z {z_after:.3} m)");
        return 1;
    }
    if feet_y(&sim) > y_before + 1.0 {
        eprintln!("FAIL player climbed over the wall");
        return 1;
    }
    0
}

/// After fragment lifetime, finite debris is gone and static colliders still match the grid.
fn blast_colliders_match_voxels_after_lifetime(config: &Config) -> i32 {
    let mut cfg = fixed_step(config);
    cfg.block_size = 0.2;
    cfg.fragment_count = 8;
    cfg.fragment_lifetime_secs = 5.0;
    cfg.blast_impulse = 4000.0;
    cfg.adhesive_concrete = 1.0e12;
    let mut sim = Simulation::new(cfg.clone());
    for x in -1..=1 {
        for z in -1..=1 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
        }
    }
    let hit = IVec3::new(0, 3, 0);
    sim.voxels
        .set(hit.x, hit.y, hit.z, Material::Concrete, true);
    let far = IVec3::new(40, 0, 0);
    sim.voxels
        .set(far.x, far.y, far.z, Material::Concrete, true);
    sim.voxels.set(40, 1, 0, Material::Concrete, true);
    sim.refresh_static();
    let far_before = sim.static_chunk_handles(far);
    let terrain_before = sim.terrain_collider_handle();
    if far_before.is_empty() {
        eprintln!("FAIL leak check far chunk had no collider");
        return 1;
    }
    sim.explode_at(sim.voxels.cell_center(hit), 0.09);
    if sim.voxels.blocks.contains_key(&hit) {
        eprintln!("FAIL leak-check cell was not removed");
        return 1;
    }
    sim.step(cfg.fixed_dt);
    if sim.fixed_collider_covers_cell(hit) {
        eprintln!("FAIL removed cell kept a fixed collider after the blast");
        return 1;
    }
    step_for(&mut sim, 5.2);
    let finite = sim.finite_debris_count();
    println!("leak_finite_debris={finite}");
    let mut failed = 0;
    if finite != 0 {
        eprintln!("FAIL {finite} finite fragments still exist after their lifetime");
        failed += 1;
    }
    if sim.static_chunk_handles(far) != far_before {
        eprintln!("FAIL far chunk handles changed while fragments expired");
        failed += 1;
    }
    if sim.terrain_collider_handle() != terrain_before {
        eprintln!("FAIL terrain handle changed while fragments expired");
        failed += 1;
    }
    if !sim.static_colliders_match_voxels() {
        eprintln!("FAIL static colliders do not match voxels after fragment lifetime");
        failed += 1;
    }
    failed
}

pub fn stress() -> i32 {
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

    let mut failures = 0;
    // A slab large enough that one blast hits hundreds of blocks.
    // The pair at x = 80 is a different collider chunk and is not connected to the slab.
    let mut sim = Simulation::new(cfg.clone());
    for x in 0..24 {
        for z in 0..24 {
            sim.voxels.set(x, 0, z, Material::Concrete, true);
            sim.voxels.set(x, 1, z, Material::Concrete, false);
        }
    }
    let far = IVec3::new(80, 0, 0);
    sim.voxels
        .set(far.x, far.y, far.z, Material::Concrete, true);
    sim.voxels.set(80, 1, 0, Material::Concrete, true);
    sim.refresh_static();
    let far_before = sim.static_chunk_handles(far);
    let terrain_before = sim.terrain_collider_handle();
    let s = sim.voxels.block_size;
    let hit = IVec3::new(
        (3.0 / s).floor() as i32,
        (0.25 / s).floor() as i32,
        (3.0 / s).floor() as i32,
    );
    if far_before.is_empty() || !sim.voxels.blocks.contains_key(&hit) {
        eprintln!("FAIL stress scene did not build the far chunk and the blast cell");
        return 1;
    }
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
        "after_6s step_ms={step_ms:.1} bodies={b2} colliders={c2} debris={} finite_debris={} rss_kb={}",
        sim.debris_count(),
        sim.finite_debris_count(),
        rss()
    );
    if sim.finite_debris_count() != 0 {
        eprintln!("FAIL finite fragments survived the first 6 s");
        failures += 1;
    }
    if sim.voxels.blocks.contains_key(&hit) {
        eprintln!("FAIL blast cell is still in the slab");
        failures += 1;
    } else if sim.fixed_collider_covers_cell(hit) {
        eprintln!("FAIL removed slab cell is still covered by a fixed collider");
        failures += 1;
    }
    if !sim.static_colliders_match_voxels() {
        eprintln!("FAIL static colliders do not match voxels after the first cycle");
        failures += 1;
    }

    // Second blast on what remains, then wait again. Growth here is a leak.
    sim.explode_at(Vec3::new(3.5, 0.25, 3.5), 2.0);
    for _ in 0..steps {
        sim.step(cfg.fixed_dt);
    }
    let (b3, c3, _) = sim.physics_stats();
    println!(
        "after_second_cycle bodies={b3} colliders={c3} debris={} finite_debris={} static_delta_colliders={} rss_kb={}",
        sim.debris_count(),
        sim.finite_debris_count(),
        c3 as i64 - c0 as i64,
        rss()
    );
    if sim.finite_debris_count() != 0 {
        eprintln!("FAIL finite fragments survived the second 6 s");
        failures += 1;
    }
    if sim.static_chunk_handles(far) != far_before {
        eprintln!("FAIL far chunk handles changed across the stress cycles");
        failures += 1;
    }
    if sim.terrain_collider_handle() != terrain_before {
        eprintln!("FAIL stress rebuilt the terrain heightfield");
        failures += 1;
    }
    if !sim.static_colliders_match_voxels() {
        eprintln!("FAIL static colliders do not match voxels after the second cycle");
        failures += 1;
    }

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
    println!(
        "detached_plate bodies={b} colliders={c} collapsed_colliders={k} rss_kb={}",
        rss()
    );
    if !sim.static_colliders_match_voxels() {
        eprintln!("FAIL detached plate left a static collider behind");
        failures += 1;
    }

    failures += villa_blast_step_budget(&cfg);

    if failures == 0 {
        println!("RESULT=PASS");
        0
    } else {
        println!("RESULT=FAIL failures={failures}");
        1
    }
}

/// One villa shot, then the steady steps. Release builds have to stay inside the step budget.
fn villa_blast_step_budget(cfg: &Config) -> i32 {
    let mut sim = Simulation::new(cfg.clone());
    crate::villa::build(&mut sim.voxels);
    sim.refresh_static();
    let terrain_before = sim.terrain_collider_handle();
    let s = cfg.block_size;
    let impact = Vec3::new((24 + 12) as f32 * s, 1.6, (28 + 40 - 2) as f32 * s);
    let t0 = std::time::Instant::now();
    sim.explode_at(impact, cfg.explosion_radius);
    println!(
        "villa_explode_ms={} villa_blocks={}",
        t0.elapsed().as_millis(),
        sim.block_count()
    );
    if sim.terrain_collider_handle() != terrain_before {
        eprintln!("FAIL villa shot rebuilt the terrain collider");
        return 1;
    }
    let mut failed = 0;
    for i in 0..20 {
        let t = std::time::Instant::now();
        sim.step(cfg.fixed_dt);
        let ms = t.elapsed().as_millis();
        if i == 0 || i == 19 {
            println!("villa_step={i} ms={ms}");
        }
        if i > 0 && !cfg!(debug_assertions) && ms >= VILLA_STEP_BUDGET_MS {
            eprintln!("FAIL villa step {i} took {ms} ms; budget is {VILLA_STEP_BUDGET_MS} ms");
            failed += 1;
            break;
        }
    }
    if !sim.static_colliders_match_voxels() {
        eprintln!("FAIL villa blast left static colliders that do not match the voxels");
        failed += 1;
    }
    failed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::villa;

    #[test]
    fn far_column_falls_after_the_anchor_is_blasted() {
        let cfg = Config::default();
        assert_eq!(distant_unsupported_falls(&cfg), 0);
        let mut small = Config::default();
        small.block_size = 0.1;
        small.gravity = 6.106645;
        assert_eq!(distant_unsupported_falls(&small), 0);
    }

    #[test]
    fn villa_blast_survives_the_next_steps() {
        let mut cfg = Config::default();
        cfg.fixed_dt = 1.0 / 120.0;
        let mut sim = Simulation::new(cfg.clone());
        villa::build(&mut sim.voxels);
        sim.refresh_static();
        let terrain_before = sim.terrain_collider_handle();
        assert!(terrain_before.is_none(), "the villa has no heightfield");
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
        assert_eq!(
            sim.terrain_collider_handle(),
            terrain_before,
            "a villa shot rebuilt the terrain collider"
        );
        println!(
            "explode_ms={} debris={} blocks={}",
            t0.elapsed().as_millis(),
            sim.debris_count(),
            sim.block_count()
        );
        let horiz = sim.mean_cluster_horizontal_speed();
        let thrown = horiz.max(sim.mean_debris_speed());
        println!(
            "cluster_horizontal_m_s={horiz:.2} debris_m_s={:.2}",
            sim.mean_debris_speed()
        );
        assert!(
            thrown > 4.0,
            "blast pieces have no speed (chunks {horiz:.2} m/s, debris {:.2} m/s)",
            sim.mean_debris_speed()
        );
        sim.take_static_meshes();
        let ti = std::time::Instant::now();
        let drawn = sim.instances().len();
        println!(
            "instances={drawn} instance_build_us={}",
            ti.elapsed().as_micros()
        );
        assert!(
            drawn < 200_000,
            "shell surface culling left {drawn} instances"
        );
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
                    ms < VILLA_STEP_BUDGET_MS,
                    "step {i} took {ms} ms; a 120 Hz step must stay under {VILLA_STEP_BUDGET_MS} ms to hold 60 FPS"
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
        assert!(
            sim.static_colliders_match_voxels(),
            "villa blast left static colliders that do not match the voxels"
        );
    }

    #[test]
    fn stacked_cubes_stay_within_a_millimetre() {
        assert_eq!(
            resting_stack_stays_within_a_millimetre(&Config::default()),
            0
        );
    }

    #[test]
    fn sideways_touch_does_not_hold_a_cube_up() {
        assert_eq!(side_contact_is_not_a_floor(&Config::default()), 0);
    }

    #[test]
    fn blast_keeps_an_untouched_chunk_collider() {
        assert_eq!(untouched_chunk_keeps_its_colliders(&Config::default()), 0);
    }

    #[test]
    fn glass_separates_and_metal_stays() {
        assert_eq!(
            glass_bond_breaks_and_metal_bond_holds(&Config::default()),
            0
        );
    }

    #[test]
    fn player_falls_when_the_floor_is_erased() {
        assert_eq!(player_falls_through_an_erased_floor(&Config::default()), 0);
    }

    #[test]
    fn player_stands_on_rubble() {
        assert_eq!(player_stands_on_settled_rubble(&Config::default()), 0);
    }

    #[test]
    fn wall_stops_the_player() {
        assert_eq!(player_is_stopped_by_a_wall(&Config::default()), 0);
    }

    #[test]
    fn fragments_expire_without_leaking_static_colliders() {
        assert_eq!(
            blast_colliders_match_voxels_after_lifetime(&Config::default()),
            0
        );
    }
}
