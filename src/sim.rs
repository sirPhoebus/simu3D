//! Simulation: one gravitational acceleration for every dynamic body, Rapier contacts, and configurable blast fragments.
//!
//! Rapier integrates gravity with symplectic Euler, which sits `0.5 * g * dt²` below the closed-form
//! free-fall position after each contact-free step. After each step, bodies with no contact are shifted
//! up by that amount. Their velocity already matches `v + a*dt`. Stacks that are touching something are
//! left to the contact solver, which needs the uncorrected step to stay stable.
//!
//! This is exact for constant gravity in free flight. It is not a general n-body solution, and it is not
//! a shockwave solver. The blast is a specific impulse (N·s/m²) with a radial falloff,
//! applied as J/m plus spin from the off-centre hit. See `blast`.

use rapier3d::control::{CharacterLength, KinematicCharacterController};
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::prelude::*;

use crate::blast;
use crate::config::Config;
use crate::material::Material;
use crate::math3::{quat_from_to, IVec3, Vec3, NEIGHBOURS};
use crate::mesh::Mesh;
use crate::split;
use crate::world::{greedy_solid_boxes, VoxelWorld};

pub struct FrameInput {
    pub strafe: f32,
    pub forward: f32,
    pub jump: bool,
    pub sprint: bool,
    pub fire: bool,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
}

impl Default for FrameInput {
    fn default() -> Self {
        Self {
            strafe: 0.0,
            forward: 0.0,
            jump: false,
            sprint: false,
            fire: false,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub center: [f32; 3],
    pub half: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub color: [f32; 4],
    pub transparent: bool,
}

#[derive(Clone, Copy)]
struct Grit {
    center: Vec3,
    half: f32,
    material: Material,
}

struct GritChunk {
    body: RigidBodyHandle,
    cubes: Vec<Grit>,
}

struct Debris {
    body: RigidBodyHandle,
    half: Vec3,
    full_size: Vec3,
    material: Material,
    age: f32,
    lifetime: f32,
}

#[derive(Clone, Copy)]
struct CollapsedBlock {
    local_center: Vec3,
    material: Material,
}

struct Collapsed {
    body: RigidBodyHandle,
    blocks: Vec<CollapsedBlock>,
    /// Small cubes this body is. Empty means draw the blocks. A lattice invented at draw time is a projection.
    cubes: Vec<Grit>,
}

struct Rocket {
    body: RigidBodyHandle,
    age: f32,
}

struct Flash {
    center: Vec3,
    age: f32,
    radius: f32,
}

pub struct Player {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub vy: f32,
    pub grounded: bool,
    jump_count: u8,
    jump_elapsed: f32,
    radius: f32,
    half_height: f32,
}

pub struct Simulation {
    pub config: Config,
    pub voxels: VoxelWorld,
    pub physics: PhysicsWorld,
    pub player: Player,
    pub chamber: u32,
    pub reserve: u32,
    pub reload_left: f32,
    pub time: f32,
    pub mesh_dirty: bool,
    static_body: RigidBodyHandle,
    chunk_colliders: std::collections::HashMap<IVec3, Vec<ColliderHandle>>,
    debris: Vec<Debris>,
    standing: Vec<Instance>,
    grit_chunks: Vec<GritChunk>,
    collapsed: Vec<Collapsed>,
    rockets: Vec<Rocket>,
    flashes: Vec<Flash>,
    probes: Vec<RigidBodyHandle>,
    input: FrameInput,
    jump_was_down: bool,
    impact_clock: f32,
}

const PLAYER_RADIUS: f32 = 0.30;
const ROCKET_FUSE: f32 = 6.0;
/// Core blocks cut into cubes. 12 × 8 = 96 bodies. The rest of the fireball is compounds.
const SPLIT_BUDGET: usize = 1;
const BODY_BUDGET: usize = 1200;
const SHELL_CLUSTER: usize = 32;
const COLLIDER_CHUNK: i32 = 16;

impl Simulation {
    pub fn new(config: Config) -> Self {
        let mut physics = PhysicsWorld::new();
        physics.gravity = Vector::new(0.0, -config.gravity, 0.0);
        physics.integration_parameters.dt = config.fixed_dt;
        physics.integration_parameters.normalized_prediction_distance = 0.02;
        // Fast bodies are sweep origins even with ccd_enabled false. That sweep is the
        // frame killer, and a stale fixed-target list panics. Rockets are ray-predicted.
        physics.integration_parameters.max_ccd_substeps = 0;
        physics.integration_parameters.num_solver_iterations = 4;

        let static_body = physics.insert_body(RigidBodyBuilder::fixed());

        let player_half_height = capsule_half_height(config.player_height_m);
        let feet_y = config.block_size + 0.05;
        let center_y = feet_y + player_half_height + PLAYER_RADIUS;
        let pos = Vec3::new(40.0, center_y, -8.0);
        // No rigid body for the player. The kinematic controller shape-casts against the world.
        // A kinematic body in this Rapier version is not given an island id, and a contact with
        // the floor then trips a debug assertion.

        let reserve = config.rocket_reserve;
        let mut sim = Self {
            config,
            voxels: VoxelWorld::new(0.0),
            physics,
            player: Player {
                pos,
                yaw: std::f32::consts::PI,
                pitch: 0.0,
                vy: 0.0,
                grounded: false,
                jump_count: 0,
                jump_elapsed: 0.0,
                radius: PLAYER_RADIUS,
                half_height: player_half_height,
            },
            chamber: 1,
            reserve,
            reload_left: 0.0,
            time: 0.0,
            mesh_dirty: true,
            static_body,
            chunk_colliders: std::collections::HashMap::new(),
            debris: Vec::new(),
            standing: Vec::new(),
            grit_chunks: Vec::new(),
            collapsed: Vec::new(),
            rockets: Vec::new(),
            flashes: Vec::new(),
            probes: Vec::new(),
            input: FrameInput::default(),
            jump_was_down: false,
            impact_clock: 0.0,
        };
        sim.voxels.block_size = sim.config.block_size;
        sim.rebuild_static_colliders();
        sim
    }

    pub fn refresh_static(&mut self) {
        self.rebuild_static_colliders();
        self.mesh_dirty = true;
    }

    /// Throw away the built house and the rubble, then build again at the current box size.
    pub fn reload_geometry(&mut self) {
        self.clear_dynamic_bodies();
        self.voxels.blocks.clear();
        self.voxels.block_size = self.config.block_size;
        let spawn = crate::villa::build(&mut self.voxels);
        self.place_player(spawn.feet, spawn.yaw, spawn.pitch);
        self.refresh_static();
    }

    fn clear_dynamic_bodies(&mut self) {
        for debris in self.debris.drain(..) {
            self.physics.remove_body(debris.body);
        }
        for chunk in self.grit_chunks.drain(..) {
            self.physics.remove_body(chunk.body);
        }
        for chunk in self.collapsed.drain(..) {
            self.physics.remove_body(chunk.body);
        }
        for rocket in self.rockets.drain(..) {
            self.physics.remove_body(rocket.body);
        }
        for probe in self.probes.drain(..) {
            self.physics.remove_body(probe);
        }
        self.flashes.clear();
    }

    pub fn place_player(&mut self, feet: Vec3, yaw: f32, pitch: f32) {
        let half_height = capsule_half_height(self.config.player_height_m);
        self.player.half_height = half_height;
        let center_y = feet.y + half_height + self.player.radius;
        self.player.pos = Vec3::new(feet.x, center_y, feet.z);
        self.player.yaw = yaw;
        self.player.pitch = pitch.clamp(-1.45, 1.45);
        self.player.vy = 0.0;
        self.player.jump_count = 0;
        self.player.jump_elapsed = 0.0;
    }

    pub fn set_input(&mut self, input: FrameInput) {
        self.input = input;
    }

    pub fn eye(&self) -> Vec3 {
        self.player.pos + Vec3::new(0.0, self.config.player_height_m * 0.5 - 0.12, 0.0)
    }

    pub fn look_direction(&self) -> Vec3 {
        let (yaw, pitch) = (self.player.yaw, self.player.pitch);
        Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        )
        .normalized()
    }

    pub fn block_count(&self) -> usize {
        self.voxels.blocks.len()
    }

    pub fn debris_count(&self) -> usize {
        self.debris.len()
    }

    pub fn grit_count(&self) -> usize {
        self.grit_chunks.iter().map(|chunk| chunk.cubes.len()).sum()
    }

    pub fn grit_lowest_y(&self) -> Option<f32> {
        let mut lowest = None;
        for chunk in &self.grit_chunks {
            let Some(body) = self.physics.bodies.get(chunk.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let q = quat_of(body.rotation());
            for cube in &chunk.cubes {
                let y = (origin + crate::math3::quat_rotate(q, cube.center)).y - cube.half;
                lowest = Some(lowest.map_or(y, |low: f32| low.min(y)));
            }
        }
        lowest
    }

    pub fn finite_debris_count(&self) -> usize {
        self.debris.iter().filter(|d| d.lifetime.is_finite()).count()
    }

    pub fn mean_debris_speed(&self) -> f32 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for debris in &self.debris {
            if let Some(body) = self.physics.bodies.get(debris.body) {
                sum += body.linvel().length();
                n += 1.0;
            }
        }
        if n == 0.0 { 0.0 } else { sum / n }
    }

    pub fn mean_cluster_horizontal_speed(&self) -> f32 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for chunk in &self.collapsed {
            if let Some(body) = self.physics.bodies.get(chunk.body) {
                let v = body.linvel();
                let w = body.angvel();
                let mut reach = self.voxels.block_size;
                for block in &chunk.blocks {
                    reach = reach.max(block.local_center.length());
                }
                sum += (v.x * v.x + v.z * v.z).sqrt() + w.length() * reach;
                n += 1.0;
            }
        }
        if n == 0.0 { 0.0 } else { sum / n }
    }

    pub fn collapsed_count(&self) -> usize {
        self.collapsed.len()
    }

    pub fn collapsed_centroid_y(&self) -> Option<f32> {
        let chunk = self.collapsed.first()?;
        Some(self.physics.bodies[chunk.body].translation().y)
    }

    pub fn highest_collapsed_y(&self) -> Option<f32> {
        self.collapsed.iter().filter_map(|chunk| {
            self.physics.bodies.get(chunk.body).map(|body| body.translation().y)
        }).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn physics_stats(&self) -> (usize, usize, usize) {
        let bodies = self.physics.bodies.iter().count();
        let colliders = self.physics.colliders.iter().count();
        let collapsed_colliders = self
            .collapsed
            .iter()
            .map(|chunk| {
                self.physics
                    .bodies
                    .get(chunk.body)
                    .map(|body| body.colliders().len())
                    .unwrap_or(0)
            })
            .sum();
        (bodies, colliders, collapsed_colliders)
    }

    pub fn spawn_probe(&mut self, center: Vec3, half: Vec3) -> RigidBodyHandle {
        let (body, _) = self.physics.insert(
            RigidBodyBuilder::dynamic()
                .translation(v_to(center))
                .linear_damping(0.0)
                .angular_damping(0.0)
                .ccd_enabled(true)
                .gravity_scale(1.0),
            ColliderBuilder::cuboid(half.x, half.y, half.z)
                .density(500.0)
                .friction(0.5)
                .restitution(0.0),
        );
        self.probes.push(body);
        body
    }

    pub fn body_translation(&self, handle: RigidBodyHandle) -> Option<Vec3> {
        self.physics.bodies.get(handle).map(|b| v_from(b.translation()))
    }

    pub fn body_linvel(&self, handle: RigidBodyHandle) -> Option<Vec3> {
        self.physics.bodies.get(handle).map(|b| v_from(b.linvel()))
    }

    /// Remove cells and run the support flood-fill. Used by the collapse test, and correct as a
    /// structural edit: those cells are gone, so anything that only stood on them must fall.
    pub fn erase_cells(&mut self, cells: &[IVec3]) {
        for cell in cells {
            self.voxels.blocks.remove(cell);
        }
        self.detach_unsupported(None);
        self.rebuild_static_colliders();
        self.mesh_dirty = true;
    }

    pub fn explode_at(&mut self, center: Vec3, radius: f32) {
        let s = self.voxels.block_size;
        let half = Vec3::new(s, s, s) * 0.5;
        let min = world_to_cell(center - Vec3::new(radius, radius, radius), s);
        let max = world_to_cell(center + Vec3::new(radius, radius, radius), s);
        let mut hit_cells = Vec::new();
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let cell = IVec3::new(x, y, z);
                    let Some(block) = self.voxels.blocks.get(&cell) else {
                        continue;
                    };
                    if block.indestructible {
                        continue;
                    }
                    if sphere_hits_aabb(center, radius, self.voxels.cell_center(cell), half) {
                        hit_cells.push(cell);
                    }
                }
            }
        }

        let face = s * s;
        let mut condemned: std::collections::HashMap<IVec3, (f32, bool)> = std::collections::HashMap::new();
        for cell in &hit_cells {
            let d = (self.voxels.cell_center(*cell) - center).length();
            let impulse = blast::face_impulse(d, radius, self.config.blast_impulse, face);
            // The whole fireball becomes rigid bodies. Only the core is cut into 10:
            // splitting every block in a 2.8 m sphere is ~14 000 bodies, and Rapier's
            // fast-body sweep then panics on a stale collider handle.
            let shatter = d <= (radius * 0.28).max(0.45);
            condemned.insert(*cell, (impulse, shatter));
        }
        let mut queue: std::collections::VecDeque<(IVec3, f32)> = condemned
            .iter()
            .map(|(cell, (impulse, _))| (*cell, *impulse))
            .collect();
        let mut seen: std::collections::HashSet<IVec3> = condemned.keys().copied().collect();
        while let Some((cell, incoming)) = queue.pop_front() {
            if condemned.len() >= 500 {
                break;
            }
            let Some(block) = self.voxels.blocks.get(&cell) else {
                continue;
            };
            let passed = incoming * block.material.shock_gain();
            for offset in NEIGHBOURS {
                let next = cell + offset;
                if seen.contains(&next) {
                    continue;
                }
                let Some(neighbour) = self.voxels.blocks.get(&next) else {
                    continue;
                };
                if neighbour.indestructible || !neighbour.material.solid() {
                    continue;
                }
                if passed <= self.config.adhesive(neighbour.material) {
                    continue;
                }
                seen.insert(next);
                condemned.insert(next, (passed, false));
                queue.push_back((next, passed));
            }
        }

        // Small cubes cannot be cut again. A later blast pushes them; it does not divide them.
        let preexisting = std::mem::take(&mut self.debris);
        let mut survivors = Vec::new();
        for debris in preexisting {
            let Some(body) = self.physics.bodies.get_mut(debris.body) else {
                continue;
            };
            let pos = v_from(body.translation());
            if sphere_hits_aabb(center, radius, pos, debris.half) {
                let dir = blast::radial_dir(center, pos, Vec3::new(0.0, 1.0, 0.0));
                let volume = debris.full_size.x * debris.full_size.y * debris.full_size.z;
                let mass = (debris.material.density() * volume).max(1.0e-6);
                let impulse = blast::face_impulse(0.0, radius, self.config.blast_impulse, debris.half.x * debris.half.y * 4.0);
                let kick = v_to(blast::ejection_velocity(dir, impulse, mass));
                body.set_linvel(body.linvel() + kick, true);
                body.wake_up(true);
            }
            survivors.push(debris);
        }
        self.debris = survivors;
        for chunk in &self.grit_chunks {
            let Some(body) = self.physics.bodies.get_mut(chunk.body) else {
                continue;
            };
            let pos = v_from(body.translation());
            if sphere_hits_aabb(center, radius, pos, Vec3::new(0.2, 0.2, 0.2)) {
                let dir = blast::radial_dir(center, pos, Vec3::new(0.0, 1.0, 0.0));
                let mass = body.mass().max(1.0e-6);
                let impulse = blast::face_impulse(0.0, radius, self.config.blast_impulse, 0.04);
                let kick = v_to(blast::ejection_velocity(dir, impulse, mass));
                body.set_linvel(body.linvel() + kick, true);
                body.wake_up(true);
            }
        }

        let mut ranked: Vec<(f32, IVec3, f32)> = Vec::new();
        for (cell, (impulse, shatter)) in &condemned {
            if !shatter {
                continue;
            }
            let at = self.voxels.cell_center(*cell);
            ranked.push(((at - center).length(), *cell, *impulse));
        }
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(SPLIT_BUDGET.max(1));
        let mut freed = Vec::new();
        for (_, cell, impulse) in ranked {
            if let Some(block) = self.voxels.blocks.remove(&cell) {
                freed.push((self.voxels.cell_center(cell), block.material, impulse, true));
            }
        }
        // Pre-existing rubble only. New hit chunks are spawned after this, and must
        // stay as the small-cube set. Taking them here turned them back into big cubes.
        let collapsed = std::mem::take(&mut self.collapsed);
        for chunk in collapsed {
            let Some(body) = self.physics.bodies.get(chunk.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let lin = body.linvel();
            let mut keep = Vec::new();
            let mut hit = Vec::new();
            for block in chunk.blocks {
                let world = origin + block.local_center;
                if sphere_hits_aabb(center, radius, world, half) {
                    hit.push((world, block.material));
                } else {
                    keep.push((world, block.material));
                }
            }
            self.physics.remove_body(chunk.body);
            let mut split_n = 0;
            let mut shell = Vec::new();
            for (world, material) in hit {
                let near = (world - center).length() <= (radius * 0.28).max(0.45);
                if near && split_n < SPLIT_BUDGET && self.debris.len() + 1000 <= BODY_BUDGET {
                    split_n += 1;
                    self.spawn_split(
                        world,
                        Vec3::new(s, s, s),
                        material,
                        lin,
                        Vector::ZERO,
                        center,
                        radius,
                        self.config.fragment_lifetime_secs,
                    );
                } else {
                    shell.push((world, material));
                }
            }
            self.spawn_shell_clusters(&shell, center, radius);
            if !keep.is_empty() {
                let (kick, spin) = self.blast_motion(&keep, center, radius);
                self.spawn_collapsed(keep, lin + v_to(kick), v_to(spin));
            }
        }
        // Every other cell in the sphere is no longer a big cube. It is the
        // 10×10×10 small cubes, one rigid chunk so the solver stays in frame.
        let mut shell_cells: Vec<(f32, IVec3, f32)> = Vec::new();
        for cell in &hit_cells {
            if !self.voxels.blocks.contains_key(cell) {
                continue;
            }
            let at = self.voxels.cell_center(*cell);
            let d = (at - center).length();
            let impulse = blast::face_impulse(d, radius, self.config.blast_impulse, face);
            shell_cells.push((d, *cell, impulse));
        }
        shell_cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let thrown: Vec<(Vec3, Material)> = shell_cells
            .into_iter()
            .filter_map(|(_, cell, _)| {
                let block = self.voxels.blocks.remove(&cell)?;
                Some((self.voxels.cell_center(cell), block.material))
            })
            .collect();
        self.spawn_shell_clusters(&thrown, center, radius);

        self.detach_unsupported(Some((center, radius)));
        // The flood-fill can detach cells well outside the fireball. A partial collider
        // rebuild leaves the old static box there, and the new body is pinned in the air.
        self.rebuild_static_colliders();
        self.spawn_freed(freed, center, radius);
        self.mesh_dirty = true;
        self.flashes.push(Flash {
            center,
            age: 0.0,
            radius,
        });
    }

    pub fn step(&mut self, dt: f32) {
        let dt = dt.max(1.0e-5);
        self.time += dt;
        self.physics.gravity = Vector::new(0.0, -self.config.gravity, 0.0);
        self.physics.integration_parameters.dt = dt;

        self.apply_look();
        self.move_player(dt);
        self.tick_reload(dt);
        if self.input.fire {
            self.try_fire();
        }
        self.input.fire = false;

        self.predict_rocket_hits(dt);
        let flight = self.snapshot_flight();
        self.physics.step();
        self.apply_analytic_freefall(&flight, dt);
        self.collect_rocket_contacts();
        self.age_debris(dt);
        self.impact_clock = (self.impact_clock - dt).max(0.0);
        self.transfer_impacts();
        self.age_flashes(dt);
        self.cull_fallen();
    }

    /// Bodies that have dropped through a hole are removed. This is not a floor.
    /// The player has no body, so the player keeps falling.
    fn cull_fallen(&mut self) {
        const Y: f32 = -40.0;
        let mut dead = Vec::new();
        for (index, debris) in self.debris.iter().enumerate() {
            if debris.lifetime.is_finite() {
                continue;
            }
            if self
                .physics
                .bodies
                .get(debris.body)
                .is_some_and(|body| body.translation().y < Y)
            {
                dead.push(index);
            }
        }
        for index in dead.into_iter().rev() {
            let debris = self.debris.swap_remove(index);
            self.physics.remove_body(debris.body);
        }
        let mut dead = Vec::new();
        for (index, chunk) in self.collapsed.iter().enumerate() {
            if self
                .physics
                .bodies
                .get(chunk.body)
                .is_some_and(|body| body.translation().y < Y)
            {
                dead.push(index);
            }
        }
        for index in dead.into_iter().rev() {
            let chunk = self.collapsed.swap_remove(index);
            self.physics.remove_body(chunk.body);
        }
        let mut dead = Vec::new();
        for (index, chunk) in self.grit_chunks.iter().enumerate() {
            if self
                .physics
                .bodies
                .get(chunk.body)
                .is_some_and(|body| body.translation().y < Y)
            {
                dead.push(index);
            }
        }
        for index in dead.into_iter().rev() {
            let chunk = self.grit_chunks.swap_remove(index);
            self.physics.remove_body(chunk.body);
        }
        let mut dead = Vec::new();
        for (index, rocket) in self.rockets.iter().enumerate() {
            if self
                .physics
                .bodies
                .get(rocket.body)
                .is_some_and(|body| body.translation().y < Y)
            {
                dead.push(index);
            }
        }
        for index in dead.into_iter().rev() {
            let rocket = self.rockets.swap_remove(index);
            self.physics.remove_body(rocket.body);
        }
    }

    pub fn take_static_meshes(&mut self) -> Option<(Mesh, Mesh)> {
        if !self.mesh_dirty {
            return None;
        }
        self.mesh_dirty = false;
        self.rebuild_standing();
        Some((Mesh::default(), Mesh::default()))
    }

    pub fn standing_gap_stale(&self) -> bool {
        let half = self.config.cube_half(self.voxels.block_size);
        self.standing
            .first()
            .is_some_and(|cube| (cube.half[0] - half).abs() > 1.0e-5)
    }

    fn rebuild_standing(&mut self) {
        self.standing.clear();
        let half = self.config.cube_half(self.voxels.block_size);
        self.standing.reserve(self.voxels.blocks.len());
        for (cell, block) in &self.voxels.blocks {
            let center = self.voxels.cell_center(*cell);
            self.standing.push(Instance {
                center: arr(center),
                half: [half, half, half],
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                color: block.material.color(),
                transparent: block.material.transparent(),
            });
        }
    }

    pub fn instances(&self) -> Vec<Instance> {
        let mut out = Vec::with_capacity(self.standing.len() + self.debris.len() + 64);
        out.extend_from_slice(&self.standing);
        for debris in &self.debris {
            let Some(body) = self.physics.bodies.get(debris.body) else {
                continue;
            };
            out.push(Instance {
                center: arr(v_from(body.translation())),
                half: [debris.half.x, debris.half.y, debris.half.z],
                rotation_xyzw: quat_of(body.rotation()),
                color: debris.material.color(),
                transparent: debris.material.transparent(),
            });
        }
        for chunk in &self.grit_chunks {
            let Some(body) = self.physics.bodies.get(chunk.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let q = quat_of(body.rotation());
            for grit in &chunk.cubes {
                let center = origin + crate::math3::quat_rotate(q, grit.center);
                out.push(Instance {
                    center: arr(center),
                    half: [grit.half, grit.half, grit.half],
                    rotation_xyzw: q,
                    color: grit.material.color(),
                    transparent: grit.material.transparent(),
                });
            }
        }
        for chunk in &self.collapsed {
            let Some(body) = self.physics.bodies.get(chunk.body) else {
                continue;
            };
            let q = quat_of(body.rotation());
            let origin = v_from(body.translation());
            if chunk.cubes.is_empty() {
                let half = self.config.cube_half(self.voxels.block_size);
                for block in &chunk.blocks {
                    let center = origin + crate::math3::quat_rotate(q, block.local_center);
                    out.push(Instance {
                        center: arr(center),
                        half: [half, half, half],
                        rotation_xyzw: q,
                        color: block.material.color(),
                        transparent: block.material.transparent(),
                    });
                }
            } else {
                for cube in &chunk.cubes {
                    let center = origin + crate::math3::quat_rotate(q, cube.center);
                    out.push(Instance {
                        center: arr(center),
                        half: [cube.half, cube.half, cube.half],
                        rotation_xyzw: q,
                        color: cube.material.color(),
                        transparent: cube.material.transparent(),
                    });
                }
            }
        }
        for rocket in &self.rockets {
            let Some(body) = self.physics.bodies.get(rocket.body) else {
                continue;
            };
            let vel = v_from(body.linvel());
            let dir = if vel.length() > 0.5 {
                vel
            } else {
                Vec3::new(0.0, 0.0, 1.0)
            };
            out.push(Instance {
                center: arr(v_from(body.translation())),
                half: [0.06, 0.06, 0.22],
                rotation_xyzw: quat_from_to(Vec3::new(0.0, 0.0, 1.0), dir),
                color: [0.18, 0.22, 0.14, 1.0],
                transparent: false,
            });
        }
        for flash in &self.flashes {
            let t = (flash.age / 0.22).clamp(0.0, 1.0);
            let extent = flash.radius * 0.35 * (1.0 - t);
            out.push(Instance {
                center: arr(flash.center),
                half: [extent, extent, extent],
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                color: [1.0, 0.55 + 0.3 * (1.0 - t), 0.15, 1.0],
                transparent: false,
            });
        }
        out
    }

    pub fn status_line(&self) -> String {
        format!(
            "t={:.1}s  yaw={:.2}  pitch={:.2}  blocks={}  debris={}  falling_structures={}  rockets={}  round {}/{}  reload {:.1}s",
            self.time,
            self.player.yaw,
            self.player.pitch,
            self.block_count(),
            self.debris_count(),
            self.collapsed_count(),
            self.rockets.len(),
            self.chamber,
            self.reserve,
            self.reload_left.max(0.0),
        )
    }

    fn apply_look(&mut self) {
        self.player.yaw += self.input.yaw_delta;
        self.player.pitch = (self.player.pitch + self.input.pitch_delta).clamp(-1.45, 1.45);
        self.input.yaw_delta = 0.0;
        self.input.pitch_delta = 0.0;
    }

    fn move_player(&mut self, dt: f32) {
        self.sync_player_height();
        let jump_pressed = self.input.jump && !self.jump_was_down;
        self.jump_was_down = self.input.jump;
        if self.player.grounded && !self.input.jump {
            self.player.jump_count = 0;
        }
        let yaw = self.player.yaw;
        let forward = Vec3::new(yaw.sin(), 0.0, -yaw.cos());
        let right = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let mut wish = forward * self.input.forward + right * self.input.strafe;
        let wish_len = wish.length();
        if wish_len > 1.0e-4 {
            wish = wish * (1.0 / wish_len);
        }
        let speed = if self.input.sprint {
            self.config.sprint_speed
        } else {
            self.config.walk_speed
        };
        let mut v0 = if self.player.grounded && !self.input.jump {
            0.0
        } else {
            self.player.vy
        };
        if self.player.grounded && self.input.jump {
            v0 = self.config.jump_speed;
            if self.player.jump_count != 1 {
                self.player.jump_count = 1;
                self.player.jump_elapsed = 0.0;
            }
        } else if jump_pressed && self.player.jump_count == 1 && self.player.jump_elapsed < 1.0 {
            v0 = self.config.jump_speed;
            self.player.jump_count = 2;
        }
        if self.player.jump_count == 1 {
            self.player.jump_elapsed += dt;
        }
        let v1 = v0 - self.config.gravity * dt;
        let dy = (v0 + v1) * 0.5 * dt;
        let displacement = wish * (speed * dt) + Vec3::new(0.0, dy, 0.0);

        let shape = Capsule::new_y(self.player.half_height, self.player.radius);
        let pose = Pose::from_translation(v_to(self.player.pos));
        let query = self.physics.query_pipeline();
        let controller = player_controller();
        let movement = controller.move_shape(dt, &query, &shape, &pose, v_to(displacement), |_| {});
        let delta = v_from(movement.translation);
        self.player.pos = self.player.pos + delta;
        self.player.grounded = movement.grounded;
        self.player.vy = if movement.grounded && v1 < 0.0 { 0.0 } else { v1 };
    }

    fn sync_player_height(&mut self) {
        let half_height = capsule_half_height(self.config.player_height_m);
        if (half_height - self.player.half_height).abs() > f32::EPSILON {
            self.player.pos.y += half_height - self.player.half_height;
            self.player.half_height = half_height;
        }
    }

    fn tick_reload(&mut self, dt: f32) {
        if self.reload_left <= 0.0 {
            return;
        }
        self.reload_left -= dt;
        if self.reload_left <= 0.0 && self.chamber == 0 && self.reserve > 0 {
            self.reserve -= 1;
            self.chamber = 1;
            self.reload_left = 0.0;
        }
    }

    fn try_fire(&mut self) {
        if self.chamber == 0 || self.reload_left > 0.0 {
            return;
        }
        self.chamber = 0;
        self.reload_left = self.config.rocket_reload_secs;
        let eye = self.eye();
        let dir = self.look_direction();
        let speed = self.config.rocket_speed.max(1.0);
        let muzzle = 1.15;
        if let Some(point) = self.sweep_ball(eye, dir * speed, muzzle / speed, None) {
            self.explode_at(point, self.config.explosion_radius);
            return;
        }
        let origin = eye + dir * muzzle;
        let velocity = dir * self.config.rocket_speed;
        let (body, _) = self.physics.insert(
            RigidBodyBuilder::dynamic()
                .translation(v_to(origin))
                .linvel(v_to(velocity))
                .linear_damping(0.0)
                .angular_damping(0.0)
                .ccd_enabled(true)
                .gravity_scale(1.0),
            ColliderBuilder::ball(0.09)
                .density(0.0)
                .restitution(0.05)
                .friction(0.2),
        );
        // Mass is set on the body so the ball's zero density does not make it weightless to impulses,
        // while gravity still accelerates it at g regardless of mass.
        self.physics.bodies[body].set_additional_mass(2.2, true);
        self.rockets.push(Rocket { body, age: 0.0 });
    }

    fn predict_rocket_hits(&mut self, dt: f32) {
        let mut hits = Vec::new();
        for (index, rocket) in self.rockets.iter().enumerate() {
            let Some(body) = self.physics.bodies.get(rocket.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let vel = v_from(body.linvel());
            let speed = vel.length();
            if speed < 0.05 {
                continue;
            }
            let dir = vel * (1.0 / speed);
            if let Some(point) = self.sweep_ball(origin, dir * speed, dt + 0.02, Some(rocket.body)) {
                hits.push((index, point));
            }
        }
        self.consume_rocket_hits(hits);
    }

    /// Sweep the rocket's own ball. A centre-line ray misses the edge of a cube,
    /// and spawning 1.15 m ahead of the eye puts the round on the far side of a thin wall.
    fn sweep_ball(
        &self,
        origin: Vec3,
        velocity: Vec3,
        max_time: f32,
        ignore: Option<RigidBodyHandle>,
    ) -> Option<Vec3> {
        let shape = Ball::new(0.11);
        let pose = Pose::from_translation(v_to(origin));
        let mut filter = QueryFilter::default();
        if let Some(body) = ignore {
            filter = filter.exclude_rigid_body(body);
        }
        let hit = self.physics.cast_shape(
            &pose,
            v_to(velocity),
            &shape,
            ShapeCastOptions {
                max_time_of_impact: max_time.max(1.0e-4),
                target_distance: 0.0,
                stop_at_penetration: true,
                compute_impact_geometry_on_penetration: false,
            },
            filter,
        )?;
        let toi = hit.1.time_of_impact;
        Some(origin + velocity * toi)
    }

    fn collect_rocket_contacts(&mut self) {
        let mut hits = Vec::new();
        for (index, rocket) in self.rockets.iter().enumerate() {
            if self.rocket_touching(rocket.body) {
                let pos = self
                    .physics
                    .bodies
                    .get(rocket.body)
                    .map(|b| v_from(b.translation()))
                    .unwrap_or(Vec3::ZERO);
                hits.push((index, pos));
            }
        }
        self.consume_rocket_hits(hits);
    }

    fn consume_rocket_hits(&mut self, mut hits: Vec<(usize, Vec3)>) {
        if hits.is_empty() {
            return;
        }
        hits.sort_by_key(|(index, _)| *index);
        hits.dedup_by_key(|(index, _)| *index);
        let radius = self.config.explosion_radius;
        for (index, point) in hits.into_iter().rev() {
            if index >= self.rockets.len() {
                continue;
            }
            let rocket = self.rockets.swap_remove(index);
            self.physics.remove_body(rocket.body);
            self.explode_at(point, radius);
        }
    }

    fn rocket_touching(&self, body: RigidBodyHandle) -> bool {
        let Some(rigid) = self.physics.bodies.get(body) else {
            return false;
        };
        for collider in rigid.colliders() {
            for pair in self.physics.narrow_phase.contact_pairs_with(*collider) {
                if pair.has_any_active_contact() {
                    return true;
                }
            }
        }
        false
    }

    fn age_debris(&mut self, dt: f32) {
        for debris in &mut self.debris {
            debris.age += dt;
        }
        let mut dead = Vec::new();
        for (index, debris) in self.debris.iter().enumerate() {
            if debris.age >= debris.lifetime {
                dead.push(index);
            }
        }
        let removed = dead.len();
        for index in dead.into_iter().rev() {
            let debris = self.debris.swap_remove(index);
            self.physics.remove_body(debris.body);
        }
        if removed >= 64 {
            trim_allocator();
        }
        let mut dead_rockets = Vec::new();
        for (index, rocket) in self.rockets.iter_mut().enumerate() {
            rocket.age += dt;
            if rocket.age >= ROCKET_FUSE {
                dead_rockets.push(index);
            }
        }
        let radius = self.config.explosion_radius;
        for index in dead_rockets.into_iter().rev() {
            let rocket = self.rockets.swap_remove(index);
            let pos = self
                .physics
                .bodies
                .get(rocket.body)
                .map(|b| v_from(b.translation()))
                .unwrap_or(Vec3::ZERO);
            self.physics.remove_body(rocket.body);
            self.explode_at(pos, radius);
        }
    }

    fn age_flashes(&mut self, dt: f32) {
        for flash in &mut self.flashes {
            flash.age += dt;
        }
        self.flashes.retain(|flash| flash.age < 0.22);
    }

    fn snapshot_flight(&self) -> Vec<(RigidBodyHandle, Vec3, Vec3)> {
        self.physics
            .bodies
            .iter()
            .filter(|(_, body)| body.is_dynamic())
            .map(|(handle, body)| (handle, v_from(body.translation()), v_from(body.linvel())))
            .collect()
    }

    /// Replace Rapier's free-flight step with the closed form.
    /// Rapier 0.36 substeps symplectic Euler internally, so a fixed post-hoc bias does not land on
    /// `y = y0 + v0 t + ½ a t²`. Overwriting contact-free bodies does. Bodies that touched something
    /// keep the contact solver's result.
    fn apply_analytic_freefall(&mut self, snapshot: &[(RigidBodyHandle, Vec3, Vec3)], dt: f32) {
        let g = self.config.gravity;
        let half_g_dt2 = 0.5 * g * dt * dt;
        for (handle, p0, v0) in snapshot {
            if self.body_has_support(*handle) {
                continue;
            }
            let Some(body) = self.physics.bodies.get_mut(*handle) else {
                continue;
            };
            if !body.is_dynamic() {
                continue;
            }
            if body.is_sleeping() {
                body.wake_up(true);
            }
            let p = Vec3::new(
                p0.x + v0.x * dt,
                p0.y + v0.y * dt - half_g_dt2,
                p0.z + v0.z * dt,
            );
            let v = Vec3::new(v0.x, v0.y - g * dt, v0.z);
            body.set_translation(v_to(p), true);
            body.set_linvel(v_to(v), true);
        }
    }

    /// A side touch is not a floor. Only a contact under the body, pushing upward, holds it.
    fn body_has_support(&self, handle: RigidBodyHandle) -> bool {
        let Some(body) = self.physics.bodies.get(handle) else {
            return false;
        };
        let body_y = body.translation().y;
        for collider in body.colliders() {
            let Some(col) = self.physics.colliders.get(*collider) else {
                continue;
            };
            for pair in self.physics.narrow_phase.contact_pairs_with(*collider) {
                let ours_first = pair.collider1 == *collider;
                for manifold in pair.manifolds() {
                    let pushing_up = if ours_first {
                        -manifold.data.normal.y
                    } else {
                        manifold.data.normal.y
                    };
                    if pushing_up <= 0.4 {
                        continue;
                    }
                    for point in manifold.points.iter() {
                        if point.dist >= 0.04 {
                            continue;
                        }
                        let local = if ours_first {
                            point.local_p1
                        } else {
                            point.local_p2
                        };
                        let world = col.position().transform_point(Vector::new(local.x, local.y, local.z));
                        if world.y < body_y - 0.02 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn overlaps_standing(&self, handle: RigidBodyHandle) -> bool {
        let Some(body) = self.physics.bodies.get(handle) else {
            return false;
        };
        let s = self.voxels.block_size.max(1.0e-4);
        for collider in body.colliders() {
            let Some(col) = self.physics.colliders.get(*collider) else {
                continue;
            };
            let aabb = col.compute_aabb();
            let min = world_to_cell(
                Vec3::new(aabb.mins.x + 1.0e-3, aabb.mins.y + 1.0e-3, aabb.mins.z + 1.0e-3),
                s,
            );
            let max = world_to_cell(
                Vec3::new(aabb.maxs.x - 1.0e-3, aabb.maxs.y - 1.0e-3, aabb.maxs.z - 1.0e-3),
                s,
            );
            for x in min.x..=max.x {
                for y in min.y..=max.y {
                    for z in min.z..=max.z {
                        if self
                            .voxels
                            .blocks
                            .get(&IVec3::new(x, y, z))
                            .is_some_and(|block| block.material.solid())
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Move a new rigid piece out of the remaining wall. A fixed nudge along the
    /// blast can push the far side of the fireball into cells that were not removed.
    /// Those pieces then sit in the wall and the contact solver holds them up.
    fn separate_from_standing(&mut self, handle: RigidBodyHandle, preferred: Vec3) {
        if !self.overlaps_standing(handle) {
            return;
        }
        let s = self.voxels.block_size.max(0.05);
        let horizontal = Vec3::new(preferred.x, 0.0, preferred.z);
        let dirs = [preferred, -preferred, horizontal, -horizontal, Vec3::Y, -Vec3::Y];
        let start = match self.physics.bodies.get(handle) {
            Some(body) => v_from(body.translation()),
            None => return,
        };
        for dir in dirs {
            if dir.length() < 1.0e-4 {
                continue;
            }
            let dir = dir.normalized();
            for step in 1..=8 {
                let Some(body) = self.physics.bodies.get_mut(handle) else {
                    return;
                };
                body.set_translation(v_to(start + dir * s * step as f32), true);
                if !self.overlaps_standing(handle) {
                    return;
                }
            }
            if let Some(body) = self.physics.bodies.get_mut(handle) {
                body.set_translation(v_to(start), true);
            }
        }
    }

    fn body_has_contact(&self, handle: RigidBodyHandle) -> bool {
        let Some(body) = self.physics.bodies.get(handle) else {
            return false;
        };
        for collider in body.colliders() {
            for pair in self.physics.narrow_phase.contact_pairs_with(*collider) {
                if pair.has_any_active_contact() {
                    return true;
                }
            }
        }
        false
    }

    fn spawn_freed(&mut self, mut freed: Vec<(Vec3, Material, f32, bool)>, center: Vec3, radius: f32) {
        freed.sort_by(|a, b| {
            let da = (a.0 - center).length();
            let db = (b.0 - center).length();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (at, material, impulse, shatter) in freed {
            if shatter {
                self.crater_parent(at, material, center, radius, impulse);
            }
        }
    }

    fn spawn_shell_clusters(&mut self, blocks: &[(Vec3, Material)], blast: Vec3, radius: f32) {
        if blocks.is_empty() {
            return;
        }
        let s = self.voxels.block_size.max(1.0e-4);
        let mut remaining: std::collections::HashMap<(i32, i32, i32), (Vec3, Material)> =
            std::collections::HashMap::new();
        for (pos, material) in blocks {
            let key = (
                (pos.x / s).round() as i32,
                (pos.y / s).round() as i32,
                (pos.z / s).round() as i32,
            );
            remaining.insert(key, (*pos, *material));
        }
        while let Some(start) = remaining.keys().next().copied() {
            let mut component = Vec::new();
            let mut stack = vec![start];
            if let Some(block) = remaining.remove(&start) {
                component.push(block);
            }
            while let Some(cell) = stack.pop() {
                if component.len() >= SHELL_CLUSTER {
                    break;
                }
                for offset in NEIGHBOURS {
                    let next = (cell.0 + offset.x, cell.1 + offset.y, cell.2 + offset.z);
                    if let Some(block) = remaining.remove(&next) {
                        component.push(block);
                        stack.push(next);
                        if component.len() >= SHELL_CLUSTER {
                            break;
                        }
                    }
                }
            }
            if component.is_empty() {
                continue;
            }
            let mut centroid = Vec3::ZERO;
            for (pos, _) in &component {
                centroid = centroid + *pos;
            }
            centroid = centroid * (1.0 / component.len() as f32);
            let dir = blast::radial_dir(blast, centroid, Vec3::new(0.0, 1.0, 0.0));
            let (kick, spin) = self.blast_motion(&component, blast, radius);
            let push = dir * (s * 2.5);
            for (pos, _) in &mut component {
                *pos = *pos + push;
            }
            self.spawn_shell_body(component, v_to(kick), v_to(spin));
            if let Some(chunk) = self.collapsed.last() {
                let handle = chunk.body;
                self.separate_from_standing(handle, dir);
            }
        }
    }

    fn crater_parent(&mut self, at: Vec3, material: Material, blast: Vec3, radius: f32, _impulse: f32) {
        let s = self.voxels.block_size;
        let size = Vec3::new(s, s, s);
        let face = s * s;
        let parent_impulse = blast::face_impulse((at - blast).length(), radius, self.config.blast_impulse, face);
        let (ejected, kept) = split::partial_break(
            size,
            blast - at,
            parent_impulse,
            self.config.adhesive(material),
            self.config.fragment_count,
            self.config.min_fragment_extent,
        );
        for piece in ejected {
            if !piece.is_cube() {
                continue;
            }
            let world = at + piece.local_center;
            let dir = blast::radial_dir(blast, world, Vec3::new(0.0, 1.0, 0.0));
            let mass = material.density() * piece.volume();
            let share = parent_impulse * (piece.volume() / size.x.max(1e-8).powi(3));
            let lin = v_to(blast::ejection_velocity(dir, share, mass));
            let ang = v_to(blast::ejection_spin(dir, share, mass, piece.half * 2.0));
            let spawned = world + dir * s;
            self.spawn_debris_box(
                spawned,
                piece.half,
                piece.half * 2.0,
                material,
                lin,
                ang,
                self.config.fragment_lifetime_secs,
            );
        }
        self.spawn_grit_chunk(at, kept, material, blast, radius);
    }

    fn spawn_grit_chunk(
        &mut self,
        at: Vec3,
        kept: Vec<split::Piece>,
        material: Material,
        blast: Vec3,
        radius: f32,
    ) {
        if kept.is_empty() {
            return;
        }
        let mut centroid = Vec3::ZERO;
        for piece in &kept {
            centroid = centroid + (at + piece.local_center);
        }
        centroid = centroid * (1.0 / kept.len() as f32);
        let worlds: Vec<(Vec3, Material)> = kept
            .iter()
            .map(|piece| (at + piece.local_center, material))
            .collect();
        let kick = self.momentum_kick(&worlds, blast, radius);
        let mut min = at + kept[0].local_center;
        let mut max = min;
        let cubes: Vec<Grit> = kept
            .into_iter()
            .map(|piece| {
                let world = at + piece.local_center;
                min = min.min(world);
                max = max.max(world);
                Grit {
                    center: world - centroid,
                    half: piece.half.x,
                    material,
                }
            })
            .collect();
        let half = cubes[0].half;
        let body = self.physics.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(v_to(centroid))
                .linvel(v_to(kick))
                .can_sleep(false)
                .ccd_enabled(false)
                .gravity_scale(1.0)
                .linear_damping(0.02)
                .angular_damping(0.05),
        );
        let volume = (half * 2.0).powi(3) * cubes.len() as f32;
        let mass = (material.density() * volume).max(1.0e-6);
        self.physics.bodies[body].set_additional_mass(mass, true);
        let local = (min + max) * 0.5 - centroid;
        let extent = (max - min) * 0.5 + Vec3::new(half, half, half);
        self.physics.colliders.insert_with_parent(
            ColliderBuilder::cuboid(extent.x, extent.y, extent.z)
                .translation(v_to(local))
                .density(0.0)
                .friction(0.45)
                .restitution(0.45)
                .restitution_combine_rule(CoefficientCombineRule::Max),
            body,
            &mut self.physics.bodies,
        );
        self.grit_chunks.push(GritChunk { body, cubes });
        let dir = blast::radial_dir(blast, centroid, Vec3::new(0.0, 1.0, 0.0));
        if let Some(rigid) = self.physics.bodies.get_mut(body) {
            let p = v_from(rigid.translation()) + dir * self.voxels.block_size;
            rigid.set_translation(v_to(p), true);
        }
        self.separate_from_standing(body, dir);
    }

    fn spawn_split(
        &mut self,
        center: Vec3,
        full_size: Vec3,
        material: Material,
        parent_linvel: Vector,
        parent_angvel: Vector,
        blast: Vec3,
        radius: f32,
        lifetime: f32,
    ) {
        let pieces = split::pieces(
            full_size,
            self.config.fragment_count,
            self.config.min_fragment_extent,
        );
        let full_volume = (full_size.x * full_size.y * full_size.z).max(1.0e-8);
        let face = self.voxels.block_size * self.voxels.block_size;
        let parent_impulse = blast::face_impulse(
            (center - blast).length(),
            radius,
            self.config.blast_impulse,
            face,
        );
        for (index, piece) in pieces.iter().copied().enumerate() {
            let world = center + piece.local_center;
            let dir = blast::radial_dir(blast, world, tumble_dir(index));
            let share = parent_impulse * (piece.volume() / full_volume);
            let mass = material.density() * piece.volume();
            let offset = v_to(piece.local_center);
            let tangential = parent_angvel.cross(offset);
            let mut lin = parent_linvel + tangential;
            lin += v_to(blast::ejection_velocity(dir, share, mass));
            let burst = if piece.local_center.length() > 1.0e-4 {
                piece.local_center.normalized()
            } else {
                tumble_dir(index)
            };
            lin += v_to(burst * 3.2);
            let gap = piece.half.x.max(piece.half.y).max(piece.half.z) * 2.4;
            let spawned_at = world + dir * 0.85 + burst * gap;
            let spin = blast::ejection_spin(dir, share, mass, piece.half * 2.0);
            self.spawn_debris_box(
                spawned_at,
                piece.half,
                piece.half * 2.0,
                material,
                lin,
                parent_angvel + v_to(spin),
                lifetime,
            );
        }
    }

    fn blast_motion(&self, blocks: &[(Vec3, Material)], blast: Vec3, radius: f32) -> (Vec3, Vec3) {
        let s = self.voxels.block_size.max(1.0e-4);
        let face = s * s;
        let mut mass = 0.0;
        let mut centroid = Vec3::ZERO;
        let mut weighted: Vec<(Vec3, f32, Vec3)> = Vec::with_capacity(blocks.len());
        for (pos, material) in blocks {
            let m = (s.powi(3) * material.density()).max(1.0e-6);
            let impulse = blast::face_impulse((*pos - blast).length(), radius, self.config.blast_impulse, face);
            let dir = blast::radial_dir(blast, *pos, Vec3::new(0.0, 1.0, 0.0));
            let vel = blast::ejection_velocity(dir, impulse, m);
            weighted.push((*pos, m, vel));
            centroid = centroid + *pos * m;
            mass += m;
        }
        if mass < 1.0e-4 {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        centroid = centroid * (1.0 / mass);
        let peak = weighted
            .iter()
            .map(|(_, _, vel)| vel.length())
            .fold(0.0_f32, f32::max);
        let cutoff = peak * 0.35;
        let mut momentum = Vec3::ZERO;
        let mut driven = 0.0;
        let mut angular = Vec3::ZERO;
        let mut ixx = 0.0;
        let mut iyy = 0.0;
        let mut izz = 0.0;
        for (pos, m, vel) in &weighted {
            if vel.length() >= cutoff {
                momentum = momentum + *vel * *m;
                driven += m;
            }
            let r = *pos - centroid;
            angular = angular + r.cross(*vel * *m);
            ixx += m * (r.y * r.y + r.z * r.z);
            iyy += m * (r.x * r.x + r.z * r.z);
            izz += m * (r.x * r.x + r.y * r.y);
        }
        let lin = if driven < 1.0e-4 {
            Vec3::ZERO
        } else {
            momentum * (1.0 / driven)
        };
        let mut spin = Vec3::new(
            angular.x / ixx.max(1.0e-4),
            angular.y / iyy.max(1.0e-4),
            angular.z / izz.max(1.0e-4),
        );
        let mag = spin.length();
        if mag > 14.0 {
            spin = spin * (14.0 / mag);
        }
        (lin, spin)
    }

    fn momentum_kick(&self, blocks: &[(Vec3, Material)], blast: Vec3, radius: f32) -> Vec3 {
        self.blast_motion(blocks, blast, radius).0
    }

    fn transfer_impacts(&mut self) {
        if self.impact_clock > 0.0 {
            return;
        }
        let mut punch = None;
        for debris in &self.debris {
            let Some(body) = self.physics.bodies.get(debris.body) else {
                continue;
            };
            let vel = v_from(body.linvel());
            let speed = vel.length();
            if speed < 6.5 || !self.body_has_contact(debris.body) {
                continue;
            }
            let mass = (debris.material.density()
                * debris.full_size.x
                * debris.full_size.y
                * debris.full_size.z)
                .max(1.0e-6);
            let dir = vel * (1.0 / speed);
            let at = v_from(body.translation()) + dir * debris.half.x.max(0.08);
            punch = Some((at, vel, mass));
            break;
        }
        let Some((at, vel, mass)) = punch else {
            return;
        };
        self.impact_clock = 0.08;
        self.impact_break(at, vel, mass);
    }

    fn impact_break(&mut self, at: Vec3, vel: Vec3, mass: f32) {
        let s = self.voxels.block_size;
        let reach = 0.55;
        let min = world_to_cell(at - Vec3::new(reach, reach, reach), s);
        let max = world_to_cell(at + Vec3::new(reach, reach, reach), s);
        let speed = vel.length().max(0.01);
        let dir = vel * (1.0 / speed);
        let share_budget = mass * speed * 0.45;
        let mut freed = Vec::new();
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    if freed.len() >= 8 {
                        break;
                    }
                    let cell = IVec3::new(x, y, z);
                    let Some(block) = self.voxels.blocks.get(&cell) else {
                        continue;
                    };
                    if block.indestructible || !block.material.solid() {
                        continue;
                    }
                    if share_budget < self.config.adhesive(block.material) {
                        continue;
                    }
                    let center = self.voxels.cell_center(cell);
                    if (center - at).length() > reach + s {
                        continue;
                    }
                    let material = block.material;
                    self.voxels.blocks.remove(&cell);
                    freed.push((center, material));
                }
            }
        }
        if freed.is_empty() {
            return;
        }
        self.detach_unsupported(None);
        self.rebuild_static_colliders();
        self.mesh_dirty = true;
        let share = share_budget / freed.len() as f32;
        let size = Vec3::new(s, s, s);
        for (center, material) in freed {
            let m = (material.density() * s * s * s).max(1.0e-6);
            self.spawn_debris_box(
                center + dir * (s * 0.35),
                size * 0.5,
                size,
                material,
                v_to(blast::ejection_velocity(dir, share, m)),
                v_to(blast::ejection_spin(dir, share, m, size)),
                f32::INFINITY,
            );
        }
    }

    fn spawn_debris_box(
        &mut self,
        center: Vec3,
        half: Vec3,
        full_size: Vec3,
        material: Material,
        linvel: Vector,
        angvel: Vector,
        lifetime: f32,
    ) {
        let phys = Vec3::new(
            (half.x - 0.003).max(0.008),
            (half.y - 0.003).max(0.008),
            (half.z - 0.003).max(0.008),
        );
        let (body, _) = self.physics.insert(
            RigidBodyBuilder::dynamic()
                .translation(v_to(center))
                .linvel(linvel)
                .angvel(angvel)
                .linear_damping(0.0)
                .angular_damping(0.12)
                .ccd_enabled(false)
                .can_sleep(false)
                .gravity_scale(1.0),
            ColliderBuilder::cuboid(phys.x, phys.y, phys.z)
                .density(material.density())
                .friction(0.45)
                .restitution(0.45)
                .restitution_combine_rule(CoefficientCombineRule::Max),
        );
        self.debris.push(Debris {
            body,
            half,
            full_size,
            material,
            age: 0.0,
            lifetime,
        });
        self.separate_from_standing(body, v_from(linvel));
    }

    fn spawn_shell_body(&mut self, world_blocks: Vec<(Vec3, Material)>, linvel: Vector, angvel: Vector) {
        if world_blocks.is_empty() {
            return;
        }
        let mut centroid = Vec3::ZERO;
        for (center, _) in &world_blocks {
            centroid = centroid + *center;
        }
        centroid = centroid * (1.0 / world_blocks.len() as f32);
        let s = self.voxels.block_size;
        let parent = Vec3::new(s, s, s);
        let pieces = split::fixed_small_cubes(parent);
        let mut pack_min = pieces[0].local_center - pieces[0].half;
        let mut pack_max = pieces[0].local_center + pieces[0].half;
        for piece in &pieces {
            pack_min = pack_min.min(piece.local_center - piece.half);
            pack_max = pack_max.max(piece.local_center + piece.half);
        }
        let pack_half = (pack_max - pack_min) * 0.5;
        let pack_center = (pack_min + pack_max) * 0.5;
        let edge = pieces[0].half.x * 2.0;
        let mut cubes_by_cell = std::collections::BTreeMap::new();
        for (center, material) in &world_blocks {
            let local_cell = *center - centroid;
            for piece in &pieces {
                let local_center = local_cell + piece.local_center;
                let world_center = centroid + local_center;
                let cell = (
                    (world_center.x / edge).round() as i32,
                    (world_center.y / edge).round() as i32,
                    (world_center.z / edge).round() as i32,
                );
                cubes_by_cell.insert(
                    cell,
                    Grit {
                        center: local_center,
                        half: piece.half.x,
                        material: *material,
                    },
                );
            }
        }
        let cubes: Vec<Grit> = cubes_by_cell
            .iter()
            .filter_map(|(&(x, y, z), cube)| {
                let enclosed = [
                    (x + 1, y, z),
                    (x - 1, y, z),
                    (x, y + 1, z),
                    (x, y - 1, z),
                    (x, y, z + 1),
                    (x, y, z - 1),
                ]
                .into_iter()
                .all(|neighbor| {
                    cubes_by_cell
                        .get(&neighbor)
                        .is_some_and(|other| other.material == cube.material)
                });
                (!enclosed).then_some(*cube)
            })
            .collect();
        let blocks: Vec<CollapsedBlock> = world_blocks
            .iter()
            .map(|(center, material)| CollapsedBlock {
                local_center: *center - centroid,
                material: *material,
            })
            .collect();
        let body = self.physics.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(v_to(centroid))
                .linvel(linvel)
                .angvel(angvel)
                .linear_damping(0.02)
                .ccd_enabled(false)
                .can_sleep(false)
                .gravity_scale(1.0),
        );
        let mass: f32 = blocks
            .iter()
            .map(|block| s.powi(3) * block.material.density())
            .sum();
        self.physics.bodies[body].set_additional_mass(mass.max(1.0e-6), true);
        for block in &blocks {
            self.physics.colliders.insert_with_parent(
                ColliderBuilder::cuboid(pack_half.x, pack_half.y, pack_half.z)
                    .translation(v_to(block.local_center + pack_center))
                    .density(0.0)
                    .friction(0.45)
                    .restitution(0.45)
                    .restitution_combine_rule(CoefficientCombineRule::Max),
                body,
                &mut self.physics.bodies,
            );
        }
        self.collapsed.push(Collapsed {
            body,
            blocks,
            cubes,
        });
    }

    fn spawn_collapsed(&mut self, world_blocks: Vec<(Vec3, Material)>, linvel: Vector, angvel: Vector) {
        if world_blocks.is_empty() {
            return;
        }
        let mut centroid = Vec3::ZERO;
        for (center, _) in &world_blocks {
            centroid = centroid + *center;
        }
        centroid = centroid * (1.0 / world_blocks.len() as f32);
        let blocks: Vec<CollapsedBlock> = world_blocks
            .iter()
            .map(|(center, material)| CollapsedBlock {
                local_center: *center - centroid,
                material: *material,
            })
            .collect();
        let body = self.physics.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(v_to(centroid))
                .linvel(linvel)
                .angvel(angvel)
                .linear_damping(0.02)
                .ccd_enabled(false)
                .can_sleep(false)
                .gravity_scale(1.0),
        );
        let mut mass = 0.0;
        let volume = self.voxels.block_size.powi(3);
        for block in &blocks {
            mass += volume * block.material.density();
        }
        self.physics.bodies[body].set_additional_mass(mass.max(1.0e-6), true);
        let boxes = compound_boxes(&blocks, self.voxels.block_size);
        let boxes = if boxes.len() <= 4 {
            boxes
        } else {
            let mut min = blocks[0].local_center;
            let mut max = blocks[0].local_center;
            for block in &blocks {
                min = min.min(block.local_center);
                max = max.max(block.local_center);
            }
            let cube = volume.cbrt() * 0.5;
            let local = (min + max) * 0.5;
            let half = (max - min) * 0.5 + Vec3::new(cube, cube, cube);
            vec![(local, half)]
        };
        for (center, half) in boxes {
            self.physics.colliders.insert_with_parent(
                ColliderBuilder::cuboid(half.x * 0.98, half.y * 0.98, half.z * 0.98)
                    .translation(v_to(center))
                    .density(0.0)
                    .friction(0.45)
                    .restitution(0.45)
                    .restitution_combine_rule(CoefficientCombineRule::Max),
                body,
                &mut self.physics.bodies,
            );
        }
        self.collapsed.push(Collapsed {
            body,
            blocks,
            cubes: Vec::new(),
        });
        let kick = v_from(linvel);
        let dir = if kick.length() > 0.1 {
            kick
        } else {
            Vec3::Y
        };
        self.separate_from_standing(body, dir);
    }

    fn detach_unsupported(&mut self, blast: Option<(Vec3, f32)>) {
        let Some((bmin, bmax)) = self.voxels.bounds() else {
            return;
        };
        let dx = (bmax.x - bmin.x + 1) as usize;
        let dy = (bmax.y - bmin.y + 1) as usize;
        let dz = (bmax.z - bmin.z + 1) as usize;
        let volume = dx.saturating_mul(dy).saturating_mul(dz);
        let mut stamp = if volume > 0 && volume <= 1_500_000 {
            vec![0u8; volume]
        } else {
            Vec::new()
        };
        let mut occupied = vec![0u8; stamp.len()];
        if !occupied.is_empty() {
            for (cell, block) in &self.voxels.blocks {
                if !block.material.solid() {
                    continue;
                }
                if let Some(i) = stamp_index(&stamp, bmin, bmax, dx, dy, *cell) {
                    occupied[i] = if block.anchored { 2 } else { 1 };
                }
            }
        }
        let mut supported_fallback = std::collections::HashSet::new();
        let mut stack: Vec<IVec3> = self
            .voxels
            .blocks
            .iter()
            .filter(|(_, block)| block.anchored && block.material.solid())
            .map(|(cell, _)| *cell)
            .collect();
        while let Some(cell) = stack.pop() {
            if let Some(i) = stamp_index(&stamp, bmin, bmax, dx, dy, cell) {
                if stamp[i] != 0 {
                    continue;
                }
            } else if !supported_fallback.insert(cell) {
                continue;
            }
            if !self
                .voxels
                .blocks
                .get(&cell)
                .is_some_and(|block| block.material.solid())
            {
                continue;
            }
            if let Some(i) = stamp_index(&stamp, bmin, bmax, dx, dy, cell) {
                stamp[i] = 1;
            }
            for offset in NEIGHBOURS {
                let next = cell + offset;
                if let Some(i) = stamp_index(&occupied, bmin, bmax, dx, dy, next) {
                    if occupied[i] != 0 && stamp[i] == 0 {
                        stack.push(next);
                    }
                } else if self.voxels.blocks.contains_key(&next)
                    && !supported_fallback.contains(&next)
                {
                    stack.push(next);
                }
            }
        }

        let mut remaining: std::collections::HashSet<IVec3> = self
            .voxels
            .blocks
            .iter()
            .filter(|(cell, block)| {
                if !block.material.solid() {
                    return false;
                }
                if let Some(i) = stamp_index(&stamp, bmin, bmax, dx, dy, **cell) {
                    stamp[i] != 1
                } else {
                    !supported_fallback.contains(*cell)
                }
            })
            .map(|(cell, _)| *cell)
            .collect();
        let mut components = Vec::new();
        while let Some(start) = remaining.iter().next().copied() {
            remaining.remove(&start);
            let mut component = Vec::new();
            let mut stack = vec![start];
            while let Some(cell) = stack.pop() {
                component.push(cell);
                for offset in NEIGHBOURS {
                    let next = cell + offset;
                    if remaining.remove(&next) {
                        stack.push(next);
                    }
                }
            }
            components.push(component);
        }
        for component in components {
            let mut world_blocks: Vec<(Vec3, Material)> = component
                .iter()
                .filter_map(|cell| {
                    let block = self.voxels.blocks.remove(cell)?;
                    Some((self.voxels.cell_center(*cell), block.material))
                })
                .collect();
            let (linvel, angvel) = if let Some((blast, radius)) = blast {
                if world_blocks.is_empty() {
                    (Vector::ZERO, Vector::ZERO)
                } else {
                    let mut centroid = Vec3::ZERO;
                    for (center, _) in &world_blocks {
                        centroid = centroid + *center;
                    }
                    centroid = centroid * (1.0 / world_blocks.len() as f32);
                    let dir = blast::radial_dir(blast, centroid, Vec3::new(0.0, 1.0, 0.0));
                    let (kick, spin) = self.blast_motion(&world_blocks, blast, radius);
                    let clear = dir * (self.voxels.block_size * 2.5);
                    for (center, _) in &mut world_blocks {
                        *center = *center + clear;
                    }
                    (v_to(kick), v_to(spin))
                }
            } else {
                (Vector::ZERO, Vector::ZERO)
            };
            self.spawn_collapsed(world_blocks, linvel, angvel);
        }
    }

    fn rebuild_static_colliders(&mut self) {
        for handles in self.chunk_colliders.values() {
            for handle in handles {
                self.physics.remove_collider(*handle);
            }
        }
        self.chunk_colliders.clear();
        let mut seen = std::collections::HashSet::new();
        for cell in self.voxels.blocks.keys() {
            seen.insert(chunk_of(*cell));
        }
        for key in seen {
            let handles = self.insert_chunk_colliders(key);
            if !handles.is_empty() {
                self.chunk_colliders.insert(key, handles);
            }
        }
        self.physics.ccd_solver.invalidate_fixed_targets_cache();
    }

    fn insert_chunk_colliders(&mut self, key: IVec3) -> Vec<ColliderHandle> {
        let origin = IVec3::new(
            key.x * COLLIDER_CHUNK,
            key.y * COLLIDER_CHUNK,
            key.z * COLLIDER_CHUNK,
        );
        let mut local = VoxelWorld::new(self.voxels.block_size);
        for x in 0..COLLIDER_CHUNK {
            for y in 0..COLLIDER_CHUNK {
                for z in 0..COLLIDER_CHUNK {
                    let cell = IVec3::new(origin.x + x, origin.y + y, origin.z + z);
                    if let Some(block) = self.voxels.blocks.get(&cell) {
                        if block.material.solid() {
                            local.blocks.insert(cell, *block);
                        }
                    }
                }
            }
        }
        let mut handles = Vec::new();
        let s = self.voxels.block_size;
        for greedy in greedy_solid_boxes(&local) {
            let size = Vec3::new(
                greedy.size.x as f32 * s,
                greedy.size.y as f32 * s,
                greedy.size.z as f32 * s,
            );
            let min = self.voxels.cell_min(greedy.min);
            let center = min + size * 0.5;
            let collider = self.physics.colliders.insert_with_parent(
                ColliderBuilder::cuboid(size.x * 0.5, size.y * 0.5, size.z * 0.5)
                    .translation(v_to(center))
                    .friction(0.5)
                    .restitution(0.4)
                    .restitution_combine_rule(CoefficientCombineRule::Max),
                self.static_body,
                &mut self.physics.bodies,
            );
            handles.push(collider);
        }
        handles
    }
}

fn stamp_index(
    stamp: &[u8],
    bmin: IVec3,
    bmax: IVec3,
    dx: usize,
    dy: usize,
    c: IVec3,
) -> Option<usize> {
    if stamp.is_empty()
        || c.x < bmin.x
        || c.y < bmin.y
        || c.z < bmin.z
        || c.x > bmax.x
        || c.y > bmax.y
        || c.z > bmax.z
    {
        return None;
    }
    let x = (c.x - bmin.x) as usize;
    let y = (c.y - bmin.y) as usize;
    let z = (c.z - bmin.z) as usize;
    Some(x + dx * (y + dy * z))
}

fn chunk_of(cell: IVec3) -> IVec3 {
    IVec3::new(
        div_floor(cell.x, COLLIDER_CHUNK),
        div_floor(cell.y, COLLIDER_CHUNK),
        div_floor(cell.z, COLLIDER_CHUNK),
    )
}

fn div_floor(a: i32, b: i32) -> i32 {
    let mut d = a / b;
    if a % b != 0 && a < 0 {
        d -= 1;
    }
    d
}

fn trim_allocator() {
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        malloc_trim(0);
    }
}

fn compound_boxes(blocks: &[CollapsedBlock], block_size: f32) -> Vec<(Vec3, Vec3)> {
    use std::collections::BTreeSet;
    let mut cells: BTreeSet<(i32, i32, i32)> = BTreeSet::new();
    for block in blocks {
        let ix = (block.local_center.x / block_size).round() as i32;
        let iy = (block.local_center.y / block_size).round() as i32;
        let iz = (block.local_center.z / block_size).round() as i32;
        cells.insert((ix, iy, iz));
    }
    let mut out = Vec::new();
    while let Some(start) = cells.iter().next().copied() {
        let mut sx = 1;
        while cells.contains(&(start.0 + sx, start.1, start.2)) {
            sx += 1;
        }
        let mut sz = 1;
        'grow_z: loop {
            for x in 0..sx {
                if !cells.contains(&(start.0 + x, start.1, start.2 + sz)) {
                    break 'grow_z;
                }
            }
            sz += 1;
        }
        let mut sy = 1;
        'grow_y: loop {
            for x in 0..sx {
                for z in 0..sz {
                    if !cells.contains(&(start.0 + x, start.1 + sy, start.2 + z)) {
                        break 'grow_y;
                    }
                }
            }
            sy += 1;
        }
        for y in 0..sy {
            for x in 0..sx {
                for z in 0..sz {
                    cells.remove(&(start.0 + x, start.1 + y, start.2 + z));
                }
            }
        }
        let center = Vec3::new(
            (start.0 as f32 + sx as f32 * 0.5) * block_size,
            (start.1 as f32 + sy as f32 * 0.5) * block_size,
            (start.2 as f32 + sz as f32 * 0.5) * block_size,
        );
        let half = Vec3::new(
            sx as f32 * block_size * 0.5,
            sy as f32 * block_size * 0.5,
            sz as f32 * block_size * 0.5,
        );
        out.push((center, half));
    }
    if out.is_empty() {
        out.push((Vec3::ZERO, Vec3::new(block_size, block_size, block_size) * 0.5));
    }
    out
}

fn player_controller() -> KinematicCharacterController {
    let mut controller = KinematicCharacterController::default();
    controller.snap_to_ground = Some(CharacterLength::Absolute(0.18));
    controller.offset = CharacterLength::Absolute(0.02);
    controller.max_slope_climb_angle = 0.85;
    controller
}

fn capsule_half_height(height: f32) -> f32 {
    (height * 0.5 - PLAYER_RADIUS).max(0.01)
}

fn tumble_dir(index: usize) -> Vec3 {
    let n = (index as i32)
        .wrapping_mul(1_103_515_245)
        .wrapping_add(12_345);
    let x = ((n & 1023) as f32 / 511.5) - 1.0;
    let y = (((n >> 10) & 1023) as f32 / 511.5) - 1.0;
    let z = (((n >> 20) & 1023) as f32 / 511.5) - 1.0;
    Vec3::new(x, y, z).normalized()
}

fn sphere_hits_aabb(center: Vec3, radius: f32, box_center: Vec3, half: Vec3) -> bool {
    let closest = center.clamp(box_center - half, box_center + half);
    let d = center - closest;
    d.dot(d) <= radius * radius
}

fn world_to_cell(p: Vec3, block_size: f32) -> IVec3 {
    IVec3::new(
        (p.x / block_size).floor() as i32,
        (p.y / block_size).floor() as i32,
        (p.z / block_size).floor() as i32,
    )
}

fn v_to(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

fn v_from(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn quat_of(rotation: &Rotation) -> [f32; 4] {
    [rotation.x, rotation.y, rotation.z, rotation.w]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_jump_requires_a_new_press_within_one_second() {
        let mut sim = Simulation::new(Config::default());
        let dt = sim.config.fixed_dt;
        sim.player.grounded = true;

        sim.input.jump = true;
        sim.move_player(dt);
        assert_eq!(sim.player.jump_count, 1);

        sim.input.jump = false;
        for _ in 0..30 {
            sim.move_player(dt);
        }
        sim.input.jump = true;
        sim.move_player(dt);
        assert_eq!(sim.player.jump_count, 2);

        let velocity_after_double_jump = sim.player.vy;
        sim.move_player(dt);
        assert!(sim.player.vy < velocity_after_double_jump);
        assert_eq!(sim.player.jump_count, 2);
    }

    #[test]
    fn second_jump_expires_after_one_second() {
        let mut sim = Simulation::new(Config::default());
        let dt = sim.config.fixed_dt;
        sim.player.grounded = true;

        sim.input.jump = true;
        sim.move_player(dt);
        sim.input.jump = false;
        for _ in 0..120 {
            sim.move_player(dt);
        }
        sim.input.jump = true;
        sim.move_player(dt);

        assert_eq!(sim.player.jump_count, 1);
    }

    #[test]
    fn held_jump_still_jumps_when_ground_contact_arrives() {
        let mut sim = Simulation::new(Config::default());
        let dt = sim.config.fixed_dt;
        sim.input.jump = true;

        sim.move_player(dt);
        assert_eq!(sim.player.jump_count, 0);

        sim.player.grounded = true;
        sim.move_player(dt);
        assert_eq!(sim.player.jump_count, 1);
        assert!(sim.player.vy > 0.0);
    }

    #[test]
    fn configured_player_height_sets_capsule_and_eye_height() {
        let config = Config {
            player_height_m: 1.5,
            ..Config::default()
        };
        let mut sim = Simulation::new(config);
        sim.place_player(Vec3::new(0.0, 2.0, 0.0), 0.0, 0.0);

        assert!((sim.player.pos.y - 2.75).abs() < 1.0e-6);
        assert!((sim.eye().y - 3.38).abs() < 1.0e-6);

        sim.config.player_height_m = 1.7;
        sim.sync_player_height();
        assert!((sim.player.pos.y - 2.85).abs() < 1.0e-6);
    }

    #[test]
    fn probe_villa_shot_hanging_cubes() {
        let mut cfg = Config::default();
        cfg.block_size = 0.1;
        cfg.explosion_radius = 2.0;
        cfg.blast_impulse = 7400.0;
        cfg.adhesive_stone = 300.0;
        cfg.adhesive_glass = 1.0;
        cfg.fragment_lifetime_secs = 0.5;
        cfg.fixed_dt = 1.0 / 120.0;
        let mut sim = Simulation::new(cfg.clone());
        sim.voxels.block_size = cfg.block_size;
        let spawn = crate::villa::build(&mut sim.voxels);
        sim.refresh_static();
        let eye = Vec3::new(spawn.feet.x, spawn.feet.y + 1.6, spawn.feet.z);
        let look = Vec3::new(spawn.yaw.sin(), -0.05, -spawn.yaw.cos()).normalized();
        let mut hit = None;
        for step in 1..80 {
            let p = eye + look * (step as f32 * 0.1);
            let cell = world_to_cell(p, cfg.block_size);
            if sim.voxels.blocks.contains_key(&cell) {
                hit = Some(sim.voxels.cell_center(cell));
                break;
            }
        }
        let center = hit.expect("look ray missed the villa");
        println!("shot at {center:?} from eye {eye:?}");
        sim.explode_at(center, cfg.explosion_radius);
        let before: Vec<(RigidBodyHandle, f32)> = sim
            .physics
            .bodies
            .iter()
            .filter(|(_, body)| body.is_dynamic())
            .map(|(h, body)| (h, body.translation().y))
            .collect();
        println!(
            "standing={} grit={} collapsed={} debris={} dynamic={}",
            sim.block_count(),
            sim.grit_count(),
            sim.collapsed_count(),
            sim.debris_count(),
            before.len()
        );
        let steps = (0.4 / cfg.fixed_dt) as i32;
        for _ in 0..steps {
            sim.step(cfg.fixed_dt);
        }
        let mut hung = 0;
        let mut fell = 0;
        for (handle, y0) in &before {
            let Some(body) = sim.physics.bodies.get(*handle) else {
                continue;
            };
            if !body.is_dynamic() {
                continue;
            }
            let y1 = body.translation().y;
            let drop = y0 - y1;
            if *y0 > 0.25 && drop < 0.15 {
                hung += 1;
                if hung <= 12 {
                    println!(
                        "hung y0={y0:.3} y1={y1:.3} drop={drop:.3} vy={:.3} support={} contact={}",
                        body.linvel().y,
                        sim.body_has_support(*handle),
                        sim.body_has_contact(*handle)
                    );
                }
            } else if drop >= 0.15 {
                fell += 1;
            }
        }
        println!("hung={hung} fell={fell}");
    }
}
