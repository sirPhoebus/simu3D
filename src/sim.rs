//! Simulation: one gravitational acceleration for every dynamic body, Rapier contacts, and configurable blast fragments.
//!
//! Rapier integrates gravity with symplectic Euler, which does not land on
//! `y = y0 + v0 t − ½ g t²`. After each step, a body that is not resting on a floor is overwritten
//! with that closed form. A ballistic step that would cross a solid voxel floor stops on that
//! surface, so a coarse `dt` cannot drop rubble through the street. A body that rests on a fixed
//! floor, or on a stack that does, keeps the contact solver's result.
//!
//! This is exact for constant gravity in free flight. It is not a general n-body solution, and it is not
//! a shockwave solver. The blast is a specific impulse (N·s/m²) with a radial falloff,
//! applied as J/m plus spin from the off-centre hit. See `blast`.

use rapier3d::control::{CharacterLength, KinematicCharacterController};
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::prelude::*;

use crate::biome::Terrain;
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
    age: f32,
    /// Glass grit is removed after this. Other grit stays.
    lifetime: f32,
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
    age: f32,
    /// Glass rubble is removed after this. Other rubble stays.
    lifetime: f32,
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

struct SolidHit {
    center: Vec3,
    size: Vec3,
    material: Material,
    linvel: Vector,
    angvel: Vector,
    lifetime: f32,
}

struct LooseHit {
    center: Vec3,
    size: Vec3,
    pieces: Vec<split::Piece>,
    material: Material,
    linvel: Vector,
    angvel: Vector,
    lifetime: f32,
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
    /// The heightfield mesh is huge and does not change when a block breaks.
    terrain_mesh_dirty: bool,
    terrain: Option<Terrain>,
    terrain_collider: Option<ColliderHandle>,
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
/// Broken glass is gone after this. It does not stay as blocks or as a bulk on the ground.
const GLASS_GONE_SECS: f32 = 2.0;
/// Walk and sprint speeds are written for this height. Other heights scale that distance.
const STRIDE_REFERENCE_M: f32 = 1.7;
const ROCKET_FUSE: f32 = 6.0;
const BODY_BUDGET: usize = 1200;
/// New rigid pieces one blast may spawn. Every hit block is tiled in full inside this cap,
/// so a crowded shot uses coarser tiles instead of leaving most of a block fused.
const RUBBLE_BREAK_SLOTS: usize = 350;
const SHELL_CLUSTER: usize = 32;
const COLLIDER_CHUNK: i32 = 16;

impl Simulation {
    pub fn new(config: Config) -> Self {
        let mut physics = PhysicsWorld::new();
        physics.gravity = Vector::new(0.0, -config.gravity, 0.0);
        physics.integration_parameters.dt = config.fixed_dt;
        physics
            .integration_parameters
            .normalized_prediction_distance = 0.02;
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
            terrain_mesh_dirty: true,
            terrain: None,
            terrain_collider: None,
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
        self.terrain_mesh_dirty = true;
    }

    /// Throw away the built house and the rubble, then build again at the current box size.
    pub fn reload_geometry(&mut self) {
        self.clear_dynamic_bodies();
        self.voxels.blocks.clear();
        self.voxels.block_size = self.config.block_size;
        let (spawn, terrain) = crate::level::build(self.config.level, &mut self.voxels);
        self.terrain = terrain;
        self.place_player(spawn.feet, spawn.yaw, spawn.pitch);
        self.refresh_static();
    }

    pub fn ground_height(&self, x: f32, z: f32) -> Option<f32> {
        self.terrain.as_ref().map(|terrain| terrain.height_at(x, z))
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
        // Feet come from the capsule already in the world, then the eyes sit
        // just under the top of the height setting. That moves the camera
        // the moment HEIGHT changes, before the next physics step resizes it.
        let feet = self.feet_y();
        Vec3::new(
            self.player.pos.x,
            feet + self.config.player_height_m - 0.12,
            self.player.pos.z,
        )
    }

    /// Soles of the capsule.
    pub fn feet_y(&self) -> f32 {
        self.player.pos.y - (self.player.half_height + self.player.radius)
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
        self.debris
            .iter()
            .filter(|d| d.lifetime.is_finite())
            .count()
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
        if n == 0.0 {
            0.0
        } else {
            sum / n
        }
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
        if n == 0.0 {
            0.0
        } else {
            sum / n
        }
    }

    pub fn collapsed_count(&self) -> usize {
        self.collapsed.len()
    }

    pub fn collapsed_centroid_y(&self) -> Option<f32> {
        let chunk = self.collapsed.first()?;
        Some(self.physics.bodies[chunk.body].translation().y)
    }

    pub fn highest_collapsed_y(&self) -> Option<f32> {
        self.collapsed
            .iter()
            .filter_map(|chunk| {
                self.physics
                    .bodies
                    .get(chunk.body)
                    .map(|body| body.translation().y)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Collider handles for the chunk that owns `cell`. Empty if that chunk has no solid cells.
    pub fn static_chunk_handles(&self, cell: IVec3) -> Vec<ColliderHandle> {
        self.chunk_colliders
            .get(&chunk_of(cell))
            .cloned()
            .unwrap_or_default()
    }

    pub fn terrain_collider_handle(&self) -> Option<ColliderHandle> {
        self.terrain_collider
    }

    /// Chunk colliders plus the terrain heightfield. Dynamic rubble is not included.
    pub fn static_collider_count(&self) -> usize {
        self.chunk_colliders
            .values()
            .map(|handles| handles.len())
            .sum::<usize>()
            + usize::from(self.terrain_collider.is_some())
    }

    /// True when every occupied chunk has colliders, no empty chunk keeps one, and every handle is live.
    pub fn static_colliders_match_voxels(&self) -> bool {
        let mut occupied = std::collections::HashSet::new();
        for (cell, block) in &self.voxels.blocks {
            if block.material.solid() {
                occupied.insert(chunk_of(*cell));
            }
        }
        if self
            .chunk_colliders
            .keys()
            .any(|key| !occupied.contains(key))
        {
            return false;
        }
        if occupied.iter().any(|key| {
            self.chunk_colliders
                .get(key)
                .is_none_or(|handles| handles.is_empty())
        }) {
            return false;
        }
        self.chunk_colliders
            .values()
            .flatten()
            .all(|handle| self.physics.colliders.get(*handle).is_some())
    }

    /// A fixed collider still covers this cell. Dynamic rubble does not count.
    pub fn fixed_collider_covers_cell(&self, cell: IVec3) -> bool {
        let center = self.voxels.cell_center(cell);
        let half = self.voxels.block_size * 0.2;
        let shape = Cuboid::new(Vector::new(half, half, half));
        let pose = Pose::from_translation(v_to(center));
        let query = self
            .physics
            .query_pipeline_with_filter(QueryFilter::only_fixed());
        let covered = query.intersect_shape(pose, &shape).next().is_some();
        covered
    }

    /// Highest point of the first fallen structure, in world metres.
    pub fn collapsed_top_y(&self) -> Option<f32> {
        let chunk = self.collapsed.first()?;
        let body = self.physics.bodies.get(chunk.body)?;
        let origin = v_from(body.translation());
        let rotation = quat_of(body.rotation());
        let half = self.voxels.block_size * 0.5;
        let mut top = f32::MIN;
        for block in &chunk.blocks {
            let y = (origin + crate::math3::quat_rotate(rotation, block.local_center)).y + half;
            top = top.max(y);
        }
        Some(top)
    }

    pub fn collapsed_vertical_speed(&self) -> Option<f32> {
        let chunk = self.collapsed.first()?;
        self.physics
            .bodies
            .get(chunk.body)
            .map(|body| body.linvel().y)
    }

    /// Centre of mass of the first fallen structure.
    pub fn collapsed_origin(&self) -> Option<Vec3> {
        let chunk = self.collapsed.first()?;
        self.physics
            .bodies
            .get(chunk.body)
            .map(|body| v_from(body.translation()))
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
        self.physics
            .bodies
            .get(handle)
            .map(|b| v_from(b.translation()))
    }

    pub fn body_linvel(&self, handle: RigidBodyHandle) -> Option<Vec3> {
        self.physics.bodies.get(handle).map(|b| v_from(b.linvel()))
    }

    /// Remove cells and run the support flood-fill. Used by the collapse test, and correct as a
    /// structural edit: those cells are gone, so anything that only stood on them must fall.
    pub fn erase_cells(&mut self, cells: &[IVec3]) {
        let mut dirty = std::collections::HashSet::new();
        for cell in cells {
            dirty.insert(chunk_of(*cell));
            self.voxels.blocks.remove(cell);
        }
        for cell in self.detach_unsupported(None) {
            dirty.insert(chunk_of(cell));
        }
        self.rebuild_chunk_keys(&dirty);
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
        let mut condemned: std::collections::HashMap<IVec3, (f32, bool)> =
            std::collections::HashMap::new();
        for cell in &hit_cells {
            let d = (self.voxels.cell_center(*cell) - center).length();
            let impulse = blast::face_impulse(d, radius, self.config.blast_impulse, face);
            // The whole fireball becomes rigid bodies. Only the core is cut into 10:
            // splitting every block in a 2.8 m sphere is ~14 000 bodies, and Rapier's
            // fast-body sweep then panics on a stale collider handle.
            let shatter = d <= (radius * 0.28).max(0.45);
            condemned.insert(*cell, (impulse, shatter));
        }
        let mut seeds: Vec<(IVec3, f32)> = condemned
            .iter()
            .map(|(cell, (impulse, _))| (*cell, *impulse))
            .collect();
        seeds.sort_by_key(|(cell, _)| (cell.x, cell.y, cell.z));
        let mut queue: std::collections::VecDeque<(IVec3, f32)> = seeds.into();
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
        // Every block the blast touches is replaced by smaller blocks that fill its
        // whole volume. A fused remainder is what left most of a block sitting there.
        // Pieces spawned here are not fed back into this same blast.
        let preexisting = std::mem::take(&mut self.debris);
        let budget = RUBBLE_BREAK_SLOTS.min(BODY_BUDGET.saturating_sub(preexisting.len()));
        let mut solid_hits = self.take_fallen_hits(center, radius);
        let mut survivors = Vec::new();
        for debris in preexisting {
            let (pos, lin, ang, hit) = {
                let Some(body) = self.physics.bodies.get(debris.body) else {
                    continue;
                };
                let pos = v_from(body.translation());
                let lin = body.linvel();
                let ang = body.angvel();
                let reach = debris.half.x.max(debris.half.y).max(debris.half.z);
                let hit = sphere_hits_aabb(center, radius, pos, Vec3::new(reach, reach, reach));
                (pos, lin, ang, hit)
            };
            if hit && self.breaks_smaller(debris.full_size) {
                self.physics.remove_body(debris.body);
                solid_hits.push(SolidHit {
                    center: pos,
                    size: debris.full_size,
                    material: debris.material,
                    linvel: lin,
                    angvel: ang,
                    lifetime: self.config.fragment_lifetime_secs,
                });
                continue;
            }
            if hit {
                let dir = blast::radial_dir(center, pos, Vec3::new(0.0, 1.0, 0.0));
                let volume = debris.full_size.x * debris.full_size.y * debris.full_size.z;
                let mass = (debris.material.density() * volume).max(1.0e-6);
                let impulse = blast::face_impulse(
                    0.0,
                    radius,
                    self.config.blast_impulse,
                    debris.half.x * debris.half.y * 4.0,
                );
                let kick = v_to(blast::ejection_velocity(dir, impulse, mass));
                let body = self.physics.bodies.get_mut(debris.body).unwrap();
                body.set_linvel(lin + kick, true);
                body.wake_up(true);
            }
            survivors.push(debris);
        }
        self.debris.append(&mut survivors);
        let loose_hits = self.take_grit_hits(center, radius);
        let mut dirty = std::collections::HashSet::new();
        let mut standing_hits = Vec::new();
        for cell in &hit_cells {
            dirty.insert(chunk_of(*cell));
            if let Some(block) = self.voxels.blocks.remove(cell) {
                standing_hits.push(SolidHit {
                    center: self.voxels.cell_center(*cell),
                    size: Vec3::new(s, s, s),
                    material: block.material,
                    linvel: Vector::new(0.0, 0.0, 0.0),
                    angvel: Vector::new(0.0, 0.0, 0.0),
                    lifetime: rubble_lifetime(block.material, self.config.fragment_lifetime_secs),
                });
            }
        }
        // A trunk or a leaf holds no bond, so the shock takes the whole tree. Each of those
        // blocks is its own body. A shell of them would leave the log in one piece.
        for cell in condemned.keys().copied().collect::<Vec<_>>() {
            let Some(block) = self.voxels.blocks.get(&cell) else {
                continue;
            };
            if !matches!(block.material, Material::Trunk | Material::Leaf) {
                continue;
            }
            dirty.insert(chunk_of(cell));
            if let Some(block) = self.voxels.blocks.remove(&cell) {
                standing_hits.push(SolidHit {
                    center: self.voxels.cell_center(cell),
                    size: Vec3::new(s, s, s),
                    material: block.material,
                    linvel: Vector::new(0.0, 0.0, 0.0),
                    angvel: Vector::new(0.0, 0.0, 0.0),
                    lifetime: GLASS_GONE_SECS,
                });
            }
        }
        let shares = fair_shares(
            solid_hits.len() + loose_hits.len() + standing_hits.len(),
            budget,
        );
        let mut share_at = 0;
        for hit in &solid_hits {
            let share = shares[share_at];
            share_at += 1;
            if !self.explode_solid(hit, share, center, radius) {
                self.spawn_whole_hit(hit, center, radius);
            }
        }
        for hit in loose_hits {
            let share = shares[share_at];
            share_at += 1;
            self.explode_loose(hit, share, center, radius);
        }
        for hit in &standing_hits {
            let share = shares[share_at];
            share_at += 1;
            if !self.explode_solid(hit, share, center, radius) {
                self.spawn_whole_hit(hit, center, radius);
            }
        }

        // Past the fireball, a bond that lost to the shock flies whole.
        let mut torn = Vec::new();
        for cell in condemned.keys().copied().collect::<Vec<_>>() {
            dirty.insert(chunk_of(cell));
            if let Some(block) = self.voxels.blocks.remove(&cell) {
                torn.push((self.voxels.cell_center(cell), block.material));
            }
        }
        self.spawn_shell_clusters(&torn, center, radius);

        for cell in self.detach_unsupported(Some((center, radius))) {
            dirty.insert(chunk_of(cell));
        }
        // Rebuild every chunk that lost a cell, including structure outside the fireball.
        // Leaving that chunk's old box in place pins the new body in the air.
        // Chunks that were not touched keep their collider handles. The terrain
        // heightfield is not part of this set.
        self.rebuild_chunk_keys(&dirty);
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
        if self.mesh_dirty {
            self.mesh_dirty = false;
            self.rebuild_standing();
        }
        // A blast dirties the voxel instances. The kilometre hill mesh stays on the GPU.
        if !self.terrain_mesh_dirty {
            return None;
        }
        self.terrain_mesh_dirty = false;
        let opaque = self
            .terrain
            .as_ref()
            .map(|terrain| terrain.mesh.clone())
            .unwrap_or_default();
        Some((opaque, Mesh::default()))
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
        let mut dynamic = self.debris.len() + self.rockets.len() + self.flashes.len();
        for chunk in &self.grit_chunks {
            dynamic += chunk.cubes.len();
        }
        for chunk in &self.collapsed {
            dynamic += if chunk.cubes.is_empty() {
                chunk.blocks.len()
            } else {
                chunk.cubes.len()
            };
        }
        let mut out = Vec::with_capacity(self.standing.len() + dynamic);
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
        let speed = (if self.input.sprint {
            self.config.sprint_speed
        } else {
            self.config.walk_speed
        }) * stride_scale(self.config.player_height_m);
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
        self.player.vy = if movement.grounded && v1 < 0.0 {
            0.0
        } else {
            v1
        };
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
            if let Some(point) = self.sweep_ball(origin, dir * speed, dt + 0.02, Some(rocket.body))
            {
                hits.push((index, point));
            }
        }
        self.consume_rocket_hits(hits);
    }

    /// Sweep the rocket's own ball. A centre-line ray misses the edge of a cube,
    /// and spawning 1.15 m ahead of the eye puts the round on the far side of a thin wall.
    /// The ball is 11 cm across, so a shot along the ground hits the lawn with its skin
    /// before the centre line reaches a trunk or a leaf. That lawn hit is not the target.
    fn sweep_ball(
        &self,
        origin: Vec3,
        velocity: Vec3,
        max_time: f32,
        ignore: Option<RigidBodyHandle>,
    ) -> Option<Vec3> {
        let speed = velocity.length();
        if speed < 1.0e-4 {
            return None;
        }
        let dir = velocity * (1.0 / speed);
        let max_time = max_time.max(1.0e-4);
        let max_dist = speed * max_time;
        let mut filter = QueryFilter::default();
        if let Some(body) = ignore {
            filter = filter.exclude_rigid_body(body);
        }
        let shape = Ball::new(0.11);
        let pose = Pose::from_translation(v_to(origin));
        let ball = self.physics.cast_shape(
            &pose,
            v_to(velocity),
            &shape,
            ShapeCastOptions {
                max_time_of_impact: max_time,
                target_distance: 0.0,
                stop_at_penetration: true,
                compute_impact_geometry_on_penetration: false,
            },
            filter,
        );
        let ray = Ray::new(v_to(origin), v_to(dir));
        let aimed = self
            .physics
            .cast_ray_and_get_normal(&ray, max_dist, true, filter);
        let ball_point = ball
            .as_ref()
            .map(|(_, hit)| origin + velocity * hit.time_of_impact);
        let aim_point = aimed
            .as_ref()
            .map(|(_, hit)| origin + dir * hit.time_of_impact);
        match (ball, aimed) {
            (None, None) => None,
            (Some(_), None) => ball_point,
            (None, Some(_)) => aim_point,
            (Some((_, ball_hit)), Some((_, aim_hit))) => {
                let ball_at = ball_point.unwrap();
                let aim_at = aim_point.unwrap();
                // normal1 points out of the swept ball. On a lawn it points up.
                let ball_contact = v_from(ball_hit.witness1);
                let ball_floor = ball_hit.normal1.y > 0.7 || self.landed_on_ground(ball_contact);
                let aim_floor = aim_hit.normal.y > 0.7 || self.landed_on_ground(aim_at);
                if ball_floor && !aim_floor {
                    Some(aim_at)
                } else if ball_floor && aim_floor {
                    self.voxel_ahead(ball_contact, dir).or(Some(ball_at))
                } else {
                    Some(ball_at)
                }
            }
        }
    }

    /// The contact is the top of a solid voxel, or the biome heightfield.
    fn landed_on_ground(&self, point: Vec3) -> bool {
        let s = self.voxels.block_size.max(1.0e-4);
        let ix = (point.x / s).floor() as i32;
        let iz = (point.z / s).floor() as i32;
        let iy = ((point.y - 0.02) / s).floor() as i32;
        if self.voxel_solid(ix, iy, iz) && !self.voxel_solid(ix, iy + 1, iz) {
            let top = (iy + 1) as f32 * s;
            if (point.y - top).abs() < 0.08 {
                return true;
            }
        }
        self.ground_height(point.x, point.z)
            .is_some_and(|height| (point.y - height).abs() < 0.12)
    }

    /// A floor hit a fraction of a block short of a trunk. The centre line stopped on the lawn.
    fn voxel_ahead(&self, point: Vec3, dir: Vec3) -> Option<Vec3> {
        let s = self.voxels.block_size.max(1.0e-4);
        let mut t = s * 0.25;
        // The ball stops about its own radius short of the face it was aimed at.
        let limit = 0.11 + s;
        while t <= limit + 1.0e-4 {
            let p = point + dir * t;
            let cell = world_to_cell(p, s);
            if let Some(block) = self.voxels.blocks.get(&cell) {
                if block.material.solid() && !block.indestructible {
                    let top = (cell.y + 1) as f32 * s;
                    let still_on_floor = (point.y - top).abs() < 0.08 && p.y <= top + 0.02;
                    if !still_on_floor {
                        return Some(self.voxels.cell_center(cell));
                    }
                }
            }
            t += s * 0.25;
        }
        None
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
        for chunk in &mut self.collapsed {
            chunk.age += dt;
        }
        let mut dead_chunks = Vec::new();
        for (index, chunk) in self.collapsed.iter().enumerate() {
            if chunk.age >= chunk.lifetime {
                dead_chunks.push(index);
            }
        }
        for index in dead_chunks.into_iter().rev() {
            let chunk = self.collapsed.swap_remove(index);
            self.physics.remove_body(chunk.body);
        }
        for chunk in &mut self.grit_chunks {
            chunk.age += dt;
        }
        let mut dead_grit = Vec::new();
        for (index, chunk) in self.grit_chunks.iter().enumerate() {
            if chunk.age >= chunk.lifetime {
                dead_grit.push(index);
            }
        }
        for index in dead_grit.into_iter().rev() {
            let chunk = self.grit_chunks.swap_remove(index);
            self.physics.remove_body(chunk.body);
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
    /// `y = y0 + v0 t − ½ g t²`. Overwriting unsupported bodies does. A body that rests on a fixed
    /// floor, or on a stack that does, keeps the contact solver's result. The overwrite is clipped
    /// to the highest solid voxel surface under the body's footprint, so one coarse step cannot
    /// put the body on the far side of that floor.
    fn apply_analytic_freefall(&mut self, snapshot: &[(RigidBodyHandle, Vec3, Vec3)], dt: f32) {
        let g = self.config.gravity;
        let half_g_dt2 = 0.5 * g * dt * dt;
        // Resolved once per step. Walking each body on its own would repeat the contact scan
        // and can leave the top of an unsupported pile behind when the piece under it is teleported.
        let supported = self.supported_bodies();
        for (handle, p0, v0) in snapshot {
            if supported.contains(handle) {
                continue;
            }
            let dy = v0.y * dt - half_g_dt2;
            let lift = self.floor_lift(*handle, *p0, dy);
            let Some(body) = self.physics.bodies.get_mut(*handle) else {
                continue;
            };
            if !body.is_dynamic() {
                continue;
            }
            if body.is_sleeping() {
                body.wake_up(true);
            }
            let mut vy = v0.y - g * dt;
            if lift > 0.0 && vy < 0.0 {
                vy = 0.0;
            }
            let p = Vec3::new(p0.x + v0.x * dt, p0.y + dy + lift, p0.z + v0.z * dt);
            let v = Vec3::new(v0.x, vy, v0.z);
            body.set_translation(v_to(p), true);
            body.set_linvel(v_to(v), true);
        }
    }

    /// How far up the ballistic step must move so the body stays on a voxel floor.
    /// Zero when the step does not meet a floor. Horizontal motion is left alone.
    fn floor_lift(&self, handle: RigidBodyHandle, p0: Vec3, dy: f32) -> f32 {
        let Some(body) = self.physics.bodies.get(handle) else {
            return 0.0;
        };
        let shift = p0 - v_from(body.translation());
        let skin = 5.0e-4;
        let mut lift = 0.0f32;
        for collider in body.colliders() {
            let Some(col) = self.physics.colliders.get(*collider) else {
                continue;
            };
            let aabb = col.compute_aabb();
            let mins = Vec3::new(
                aabb.mins.x + shift.x,
                aabb.mins.y + shift.y,
                aabb.mins.z + shift.z,
            );
            let maxs = Vec3::new(
                aabb.maxs.x + shift.x,
                aabb.maxs.y + shift.y,
                aabb.maxs.z + shift.z,
            );
            let start_bottom = mins.y;
            let dest_bottom = start_bottom + dy;
            let Some(top) = self.support_surface(mins, maxs, start_bottom, dest_bottom) else {
                continue;
            };
            let target = top + skin;
            if dest_bottom < target {
                lift = lift.max(target - dest_bottom);
            }
        }
        lift
    }

    /// Highest free voxel surface under the shrunk footprint that this vertical step meets.
    fn support_surface(
        &self,
        mins: Vec3,
        maxs: Vec3,
        start_bottom: f32,
        dest_bottom: f32,
    ) -> Option<f32> {
        // A 1 mm sideways overlap of a wall must not count as ground.
        let shrink = 0.002;
        let (x0, x1) = shrunk_span(mins.x, maxs.x, shrink);
        let (z0, z1) = shrunk_span(mins.z, maxs.z, shrink);
        let xm = (x0 + x1) * 0.5;
        let zm = (z0 + z1) * 0.5;
        let samples = [
            (x0, z0),
            (x0, z1),
            (x1, z0),
            (x1, z1),
            (x0, zm),
            (x1, zm),
            (xm, z0),
            (xm, z1),
            (xm, zm),
        ];
        let mut best: Option<f32> = None;
        for (x, z) in samples {
            let Some(top) = self.column_floor(x, z, start_bottom, dest_bottom) else {
                continue;
            };
            best = Some(best.map_or(top, |so_far| so_far.max(top)));
        }
        best
    }

    /// Top of the highest solid cell in this column with air above it, between the
    /// ballistic destination and one block above the feet. A cell with solid above
    /// it is a wall, not a floor, so a body overlapping that wall keeps falling.
    fn column_floor(&self, x: f32, z: f32, start_bottom: f32, dest_bottom: f32) -> Option<f32> {
        let s = self.voxels.block_size.max(1.0e-4);
        let ix = (x / s).floor() as i32;
        let iz = (z / s).floor() as i32;
        let t_max = start_bottom + s;
        let t_min = dest_bottom - 1.0e-3;
        if t_max < t_min {
            return None;
        }
        let mut iy = (t_max / s).floor() as i32;
        let iy_lo = (t_min / s).floor() as i32;
        let mut best: Option<f32> = None;
        let mut seen = 0;
        while iy >= iy_lo && seen < 64 {
            seen += 1;
            if self.voxel_solid(ix, iy, iz) && !self.voxel_solid(ix, iy + 1, iz) {
                let top = (iy + 1) as f32 * s;
                if top >= t_min && top <= t_max {
                    best = Some(best.map_or(top, |so_far| so_far.max(top)));
                }
            }
            if iy == i32::MIN {
                break;
            }
            iy -= 1;
        }
        best
    }

    fn voxel_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.voxels
            .blocks
            .get(&IVec3::new(x, y, z))
            .is_some_and(|block| block.material.solid())
    }

    /// Dynamic bodies that rest on a fixed floor with solid voxels under them, plus anything
    /// stacked on that set. A side touch is not a floor. Two bodies that only lean on each
    /// other are not in the set, so both take the free-fall step.
    fn supported_bodies(&self) -> std::collections::HashSet<RigidBodyHandle> {
        let mut on_floor = Vec::new();
        let mut children: std::collections::HashMap<RigidBodyHandle, Vec<RigidBodyHandle>> =
            std::collections::HashMap::new();
        for (handle, body) in self.physics.bodies.iter() {
            if !body.is_dynamic() {
                continue;
            }
            let body_y = body.translation().y;
            let body_pos = v_from(body.translation());
            let mut grounded = false;
            let mut parents = Vec::new();
            'colliders: for collider in body.colliders() {
                let Some(col) = self.physics.colliders.get(*collider) else {
                    continue;
                };
                // Cuboid contacts land on the face corners, and a fallen plate is wider
                // than one block. A point on the bottom of this collider is a floor.
                // A point on the side is not, even when the normal tilts up.
                let aabb = col.compute_aabb();
                let span_y = (aabb.maxs.y - aabb.mins.y).max(1.0e-4);
                let margin = (self.voxels.block_size * 0.05).max(0.01);
                for pair in self.physics.narrow_phase.contact_pairs_with(*collider) {
                    let ours_first = pair.collider1 == *collider;
                    let other = if ours_first {
                        pair.collider2
                    } else {
                        pair.collider1
                    };
                    for manifold in pair.manifolds() {
                        let pushing_up = if ours_first {
                            -manifold.data.normal.y
                        } else {
                            manifold.data.normal.y
                        };
                        if pushing_up <= 0.7 {
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
                            let world = col
                                .position()
                                .transform_point(Vector::new(local.x, local.y, local.z));
                            let over_footprint = world.x >= aabb.mins.x - margin
                                && world.x <= aabb.maxs.x + margin
                                && world.z >= aabb.mins.z - margin
                                && world.z <= aabb.maxs.z + margin;
                            let near_bottom = world.y <= aabb.mins.y + span_y * 0.3;
                            if world.y >= body_y - 0.02 || !over_footprint || !near_bottom {
                                continue;
                            }
                            let Some(other_handle) = self
                                .physics
                                .colliders
                                .get(other)
                                .and_then(|collider| collider.parent())
                            else {
                                continue;
                            };
                            let Some(other_body) = self.physics.bodies.get(other_handle) else {
                                continue;
                            };
                            if other_body.is_dynamic() {
                                if other_handle != handle && !parents.contains(&other_handle) {
                                    parents.push(other_handle);
                                }
                            } else if self.standing_below(body_pos) {
                                grounded = true;
                                break 'colliders;
                            }
                        }
                    }
                }
            }
            if grounded {
                on_floor.push(handle);
            } else {
                for parent in parents {
                    children.entry(parent).or_default().push(handle);
                }
            }
        }
        let mut supported = std::collections::HashSet::with_capacity(on_floor.len());
        let mut queue = on_floor;
        for handle in &queue {
            supported.insert(*handle);
        }
        let mut head = 0;
        while head < queue.len() {
            let parent = queue[head];
            head += 1;
            if let Some(above) = children.get(&parent) {
                for child in above {
                    if supported.insert(*child) {
                        queue.push(*child);
                    }
                }
            }
        }
        supported
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
                Vec3::new(
                    aabb.mins.x + 1.0e-3,
                    aabb.mins.y + 1.0e-3,
                    aabb.mins.z + 1.0e-3,
                ),
                s,
            );
            let max = world_to_cell(
                Vec3::new(
                    aabb.maxs.x - 1.0e-3,
                    aabb.maxs.y - 1.0e-3,
                    aabb.maxs.z - 1.0e-3,
                ),
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

    fn standing_below(&self, pos: Vec3) -> bool {
        let cell = world_to_cell(pos, self.voxels.block_size.max(1.0e-4));
        (1..8).any(|dy| {
            self.voxels
                .blocks
                .get(&IVec3::new(cell.x, cell.y - dy, cell.z))
                .is_some_and(|block| block.material.solid())
        })
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
        let dirs = [
            preferred,
            -preferred,
            horizontal,
            -horizontal,
            Vec3::Y,
            -Vec3::Y,
        ];
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

    fn spawn_shell_clusters(&mut self, blocks: &[(Vec3, Material)], blast: Vec3, radius: f32) {
        if blocks.is_empty() {
            return;
        }
        // Glass, trunks, and leaves are their own short-lived bodies. Left in a plaster
        // chunk they would sit on the ground.
        let mut glass = Vec::new();
        let mut rest = Vec::new();
        for block in blocks {
            if block.1.shatters() {
                glass.push(*block);
            } else {
                rest.push(*block);
            }
        }
        self.spawn_shell_groups(&rest, blast, radius, f32::INFINITY);
        self.spawn_shell_groups(&glass, blast, radius, GLASS_GONE_SECS);
    }

    fn spawn_shell_groups(
        &mut self,
        blocks: &[(Vec3, Material)],
        blast: Vec3,
        radius: f32,
        lifetime: f32,
    ) {
        if blocks.is_empty() {
            return;
        }
        let s = self.voxels.block_size.max(1.0e-4);
        let pieces = split::fixed_small_cubes(Vec3::new(s, s, s));
        let mut grid = ShellGrid::default();
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
            self.spawn_shell_body(
                component,
                v_to(kick),
                v_to(spin),
                &pieces,
                &mut grid,
                lifetime,
            );
            if let Some(chunk) = self.collapsed.last() {
                let handle = chunk.body;
                self.separate_from_standing(handle, dir);
            }
        }
    }

    /// A block that is already dynamic still cuts into smaller blocks when a blast reaches it.
    fn breaks_smaller(&self, size: Vec3) -> bool {
        let n = split::grid_n(self.config.fragment_count);
        if n <= 1 {
            return false;
        }
        let nf = n as f32;
        let cell = Vec3::new(size.x / nf, size.y / nf, size.z / nf);
        let side = cell.x.min(cell.y).min(cell.z);
        let longest = size.x.max(size.y).max(size.z);
        side >= self.config.min_fragment_extent && side + 1.0e-4 < longest
    }

    fn take_fallen_hits(&mut self, center: Vec3, radius: f32) -> Vec<SolidHit> {
        let s = self.voxels.block_size;
        let half = Vec3::new(s, s, s) * 0.5;
        let size = Vec3::new(s, s, s);
        let collapsed = std::mem::take(&mut self.collapsed);
        let mut hits = Vec::new();
        for chunk in collapsed {
            let Some(body) = self.physics.bodies.get(chunk.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let rotation = quat_of(body.rotation());
            let lin = body.linvel();
            let ang = body.angvel();
            let handle = chunk.body;
            let mut keep = Vec::new();
            let mut struck = Vec::new();
            for block in &chunk.blocks {
                let world = origin + crate::math3::quat_rotate(rotation, block.local_center);
                if sphere_hits_aabb(center, radius, world, half) {
                    struck.push((world, block.material));
                } else {
                    keep.push((world, block.material));
                }
            }
            if struck.is_empty() {
                self.collapsed.push(chunk);
                continue;
            }
            self.physics.remove_body(handle);
            for (world, material) in struck {
                let offset = v_to(world - origin);
                let carried = lin + ang.cross(offset);
                if self.breaks_smaller(size) {
                    hits.push(SolidHit {
                        center: world,
                        size,
                        material,
                        linvel: carried,
                        angvel: ang,
                        lifetime: self.config.fragment_lifetime_secs,
                    });
                } else {
                    keep.push((world, material));
                }
            }
            if !keep.is_empty() {
                let (kick, spin) = self.blast_motion(&keep, center, radius);
                self.spawn_collapsed(keep, lin + v_to(kick), ang + v_to(spin));
            }
        }
        hits
    }

    fn take_grit_hits(&mut self, center: Vec3, radius: f32) -> Vec<LooseHit> {
        let grit = std::mem::take(&mut self.grit_chunks);
        let mut hits = Vec::new();
        for chunk in grit {
            let Some(body) = self.physics.bodies.get(chunk.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let rotation = quat_of(body.rotation());
            let lin = body.linvel();
            let ang = body.angvel();
            let handle = chunk.body;
            let lifetime = chunk.lifetime;
            let mut struck = false;
            let mut worlds = Vec::new();
            for cube in &chunk.cubes {
                let world = origin + crate::math3::quat_rotate(rotation, cube.center);
                let half = Vec3::new(cube.half, cube.half, cube.half);
                if sphere_hits_aabb(center, radius, world, half) {
                    struck = true;
                }
                worlds.push((world, cube.half, cube.material));
            }
            if !struck {
                self.grit_chunks.push(chunk);
                continue;
            }
            self.physics.remove_body(handle);
            let mut by_material: Vec<(Material, Vec<(Vec3, f32)>)> = Vec::new();
            for (world, half, material) in worlds {
                if let Some((_, cubes)) = by_material.iter_mut().find(|(kind, _)| *kind == material)
                {
                    cubes.push((world, half));
                } else {
                    by_material.push((material, vec![(world, half)]));
                }
            }
            for (material, cubes) in by_material {
                let mut min = cubes[0].0 - Vec3::new(cubes[0].1, cubes[0].1, cubes[0].1);
                let mut max = cubes[0].0 + Vec3::new(cubes[0].1, cubes[0].1, cubes[0].1);
                for (world, half) in &cubes {
                    let h = Vec3::new(*half, *half, *half);
                    min = min.min(*world - h);
                    max = max.max(*world + h);
                }
                let bounds_center = (min + max) * 0.5;
                let pieces = cubes
                    .iter()
                    .map(|(world, half)| split::Piece {
                        local_center: *world - bounds_center,
                        half: Vec3::new(*half, *half, *half),
                    })
                    .collect();
                hits.push(LooseHit {
                    center: bounds_center,
                    size: max - min,
                    pieces,
                    material,
                    linvel: lin,
                    angvel: ang,
                    lifetime,
                });
            }
        }
        hits
    }

    /// Tile the whole box. `share` is how many bodies this block may use. There is no leftover parent.
    fn explode_solid(&mut self, hit: &SolidHit, share: usize, blast: Vec3, radius: f32) -> bool {
        let pieces = self.tiling_pieces(hit.size, share);
        if pieces.len() < 2 {
            return false;
        }
        for (index, piece) in pieces.iter().copied().enumerate() {
            self.spawn_fragment_piece(
                hit.center,
                hit.size,
                piece,
                index,
                hit.material,
                hit.linvel,
                hit.angvel,
                blast,
                radius,
                hit.lifetime,
            );
        }
        true
    }

    fn explode_loose(&mut self, mut hit: LooseHit, share: usize, blast: Vec3, radius: f32) {
        if hit.pieces.is_empty() {
            return;
        }
        if hit.pieces.len() == 1 {
            let piece = hit.pieces[0];
            let solid = SolidHit {
                center: hit.center + piece.local_center,
                size: piece.half * 2.0,
                material: hit.material,
                linvel: hit.linvel,
                angvel: hit.angvel,
                lifetime: hit.lifetime,
            };
            if !self.explode_solid(&solid, share, blast, radius) {
                self.spawn_whole_hit(&solid, blast, radius);
            }
            return;
        }
        if share < 2 {
            self.spawn_whole_hit(
                &SolidHit {
                    center: hit.center,
                    size: hit.size,
                    material: hit.material,
                    linvel: hit.linvel,
                    angvel: hit.angvel,
                    lifetime: hit.lifetime,
                },
                blast,
                radius,
            );
            return;
        }
        let group_count = share.min(hit.pieces.len());
        let groups = spatial_groups(std::mem::take(&mut hit.pieces), group_count);
        for (index, group) in groups.iter().enumerate() {
            self.spawn_piece_group(&hit, group, index, blast, radius);
        }
    }

    fn spawn_piece_group(
        &mut self,
        hit: &LooseHit,
        group: &[split::Piece],
        index: usize,
        blast: Vec3,
        radius: f32,
    ) {
        let mut min = hit.center + group[0].local_center - group[0].half;
        let mut max = hit.center + group[0].local_center + group[0].half;
        for piece in group {
            let world = hit.center + piece.local_center;
            min = min.min(world - piece.half);
            max = max.max(world + piece.half);
        }
        let world_center = (min + max) * 0.5;
        let piece = split::Piece {
            local_center: world_center - hit.center,
            half: (max - min) * 0.5,
        };
        self.spawn_fragment_piece(
            hit.center,
            hit.size,
            piece,
            index,
            hit.material,
            hit.linvel,
            hit.angvel,
            blast,
            radius,
            hit.lifetime,
        );
    }

    fn spawn_whole_hit(&mut self, hit: &SolidHit, blast: Vec3, radius: f32) {
        let dir = blast::radial_dir(blast, hit.center, Vec3::new(0.0, 1.0, 0.0));
        let volume = (hit.size.x * hit.size.y * hit.size.z).max(1.0e-8);
        let mass = (hit.material.density() * volume).max(1.0e-6);
        let impulse = blast::face_impulse(
            (hit.center - blast).length(),
            radius,
            self.config.blast_impulse,
            hit.size.x * hit.size.y,
        );
        let kick = v_to(blast::ejection_velocity(dir, impulse, mass));
        self.spawn_debris_box(
            hit.center,
            hit.size * 0.5,
            hit.size,
            hit.material,
            hit.linvel + kick,
            hit.angvel,
            hit.lifetime,
        );
    }

    fn tiling_pieces(&self, size: Vec3, share: usize) -> Vec<split::Piece> {
        if share < 2 {
            return Vec::new();
        }
        let min_extent = self.config.min_fragment_extent;
        let fine = split::grid_n(self.config.fragment_count).max(1);
        let shortest = size.x.min(size.y).min(size.z);
        let mut n = 1usize;
        let mut k = 2usize;
        while k <= fine && k * k * k <= share {
            if shortest / k as f32 + 1.0e-6 < min_extent {
                break;
            }
            n = k;
            k += 1;
        }
        if n >= 2 {
            let pieces = split::pieces(size, n * n * n, min_extent);
            if pieces.len() >= 2 {
                return pieces;
            }
        }
        self.slab_pieces(size, share.min(fine.max(2)))
    }

    fn slab_pieces(&self, size: Vec3, share: usize) -> Vec<split::Piece> {
        let min_extent = self.config.min_fragment_extent;
        let (axis, length) = longest_axis(size);
        let mut m = share.max(2);
        while m > 2 && length / (m as f32) < min_extent {
            m -= 1;
        }
        if length / (m as f32) < min_extent {
            return Vec::new();
        }
        let cell = length / m as f32;
        let origin = -length * 0.5;
        let mut pieces = Vec::with_capacity(m);
        for i in 0..m {
            let mut center = Vec3::ZERO;
            let mut half = size * 0.5;
            let offset = origin + cell * (i as f32 + 0.5);
            match axis {
                0 => {
                    center.x = offset;
                    half.x = cell * 0.5;
                }
                1 => {
                    center.y = offset;
                    half.y = cell * 0.5;
                }
                _ => {
                    center.z = offset;
                    half.z = cell * 0.5;
                }
            }
            pieces.push(split::Piece {
                local_center: center,
                half,
            });
        }
        pieces
    }

    fn spawn_fragment_piece(
        &mut self,
        center: Vec3,
        full_size: Vec3,
        piece: split::Piece,
        index: usize,
        material: Material,
        parent_linvel: Vector,
        parent_angvel: Vector,
        blast: Vec3,
        radius: f32,
        lifetime: f32,
    ) {
        let full_volume = (full_size.x * full_size.y * full_size.z).max(1.0e-8);
        let face = self.voxels.block_size * self.voxels.block_size;
        let parent_impulse = blast::face_impulse(
            (center - blast).length(),
            radius,
            self.config.blast_impulse,
            face,
        );
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

    fn blast_motion(&self, blocks: &[(Vec3, Material)], blast: Vec3, radius: f32) -> (Vec3, Vec3) {
        let s = self.voxels.block_size.max(1.0e-4);
        let face = s * s;
        let mut mass = 0.0;
        let mut centroid = Vec3::ZERO;
        let mut weighted: Vec<(Vec3, f32, Vec3)> = Vec::with_capacity(blocks.len());
        for (pos, material) in blocks {
            let m = (s.powi(3) * material.density()).max(1.0e-6);
            let impulse = blast::face_impulse(
                (*pos - blast).length(),
                radius,
                self.config.blast_impulse,
                face,
            );
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
        let mut dirty_cells = std::collections::HashSet::new();
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
                    dirty_cells.insert(chunk_of(cell));
                    self.voxels.blocks.remove(&cell);
                    freed.push((center, material));
                }
            }
        }
        if freed.is_empty() {
            return;
        }
        for cell in self.detach_unsupported(None) {
            dirty_cells.insert(chunk_of(cell));
        }
        self.rebuild_chunk_keys(&dirty_cells);
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
                rubble_lifetime(material, f32::INFINITY),
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
        let lifetime = rubble_lifetime(material, lifetime);
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
                .restitution_combine_rule(CoefficientCombineRule::Max)
                // Fragments still hit the world and the big pieces. They do not hit each
                // other: a few hundred overlapping dust cubes make the shot stall.
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_2,
                    Group::ALL.difference(Group::GROUP_2),
                    InteractionTestMode::And,
                )),
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

    fn spawn_shell_body(
        &mut self,
        world_blocks: Vec<(Vec3, Material)>,
        linvel: Vector,
        angvel: Vector,
        pieces: &[split::Piece],
        grid: &mut ShellGrid,
        lifetime: f32,
    ) {
        if world_blocks.is_empty() || pieces.is_empty() {
            return;
        }
        let mut centroid = Vec3::ZERO;
        for (center, _) in &world_blocks {
            centroid = centroid + *center;
        }
        centroid = centroid * (1.0 / world_blocks.len() as f32);
        let s = self.voxels.block_size;
        let cubes = grid.surface(&world_blocks, centroid, pieces);
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
        // One collider per block made a ground shot ~2000 overlapping shapes, and the
        // solver kept that cost after the flash. The merged boxes cover the same cells.
        for (center, half) in collider_boxes(&blocks, s) {
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
        let lifetime = rubble_lifetime_of(lifetime, &blocks);
        self.collapsed.push(Collapsed {
            body,
            blocks,
            cubes,
            age: 0.0,
            lifetime,
        });
    }

    fn spawn_collapsed(
        &mut self,
        world_blocks: Vec<(Vec3, Material)>,
        linvel: Vector,
        angvel: Vector,
    ) {
        if world_blocks.is_empty() {
            return;
        }
        let mut glass = Vec::new();
        let mut rest = Vec::new();
        for block in world_blocks {
            if block.1.shatters() {
                glass.push(block);
            } else {
                rest.push(block);
            }
        }
        self.spawn_collapsed_body(rest, linvel, angvel, f32::INFINITY);
        self.spawn_collapsed_body(glass, linvel, angvel, GLASS_GONE_SECS);
    }

    fn spawn_collapsed_body(
        &mut self,
        world_blocks: Vec<(Vec3, Material)>,
        linvel: Vector,
        angvel: Vector,
        lifetime: f32,
    ) {
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
        for (center, half) in collider_boxes(&blocks, self.voxels.block_size) {
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
            age: 0.0,
            lifetime,
        });
        let kick = v_from(linvel);
        let dir = if kick.length() > 0.1 { kick } else { Vec3::Y };
        self.separate_from_standing(body, dir);
    }

    fn detach_unsupported(&mut self, blast: Option<(Vec3, f32)>) -> Vec<IVec3> {
        let Some((bmin, bmax)) = self.voxels.bounds() else {
            return Vec::new();
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
        let mut removed = Vec::new();
        for component in components {
            removed.extend(component.iter().copied());
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
        removed
    }

    fn rebuild_chunk_keys(&mut self, keys: &std::collections::HashSet<IVec3>) {
        if keys.is_empty() {
            return;
        }
        for key in keys {
            if let Some(handles) = self.chunk_colliders.remove(key) {
                for handle in handles {
                    self.physics.remove_collider(handle);
                }
            }
            let handles = self.insert_chunk_colliders(*key);
            if !handles.is_empty() {
                self.chunk_colliders.insert(*key, handles);
            }
        }
        self.physics.ccd_solver.invalidate_fixed_targets_cache();
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
        self.install_terrain_collider();
        self.physics.ccd_solver.invalidate_fixed_targets_cache();
    }

    fn install_terrain_collider(&mut self) {
        if let Some(handle) = self.terrain_collider.take() {
            self.physics.remove_collider(handle);
        }
        let Some(terrain) = &self.terrain else {
            return;
        };
        let heights = rapier3d::parry::utils::Array2::new(
            terrain.samples,
            terrain.samples,
            terrain.collider_heights(),
        );
        let scale = Vector::new(terrain.width, 1.0, terrain.width);
        let origin = Vector::new(terrain.width * 0.5, 0.0, terrain.width * 0.5);
        let collider = self.physics.colliders.insert_with_parent(
            ColliderBuilder::heightfield_with_flags(
                heights,
                scale,
                rapier3d::parry::shape::HeightFieldFlags::FIX_INTERNAL_EDGES,
            )
            .translation(origin)
            .friction(0.9)
            .restitution(0.45)
            .restitution_combine_rule(CoefficientCombineRule::Max),
            self.static_body,
            &mut self.physics.bodies,
        );
        self.terrain_collider = Some(collider);
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
                    .friction(0.8)
                    .restitution(0.0),
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

/// Reused raster for one shell chunk. A hash map per small cube was the fire hitch.
struct ShellGrid {
    samples: Vec<(i32, i32, i32, Grit)>,
    occupied: Vec<u8>,
    centers: Vec<Vec3>,
    materials: Vec<Material>,
}

impl Default for ShellGrid {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            occupied: Vec::new(),
            centers: Vec::new(),
            materials: Vec::new(),
        }
    }
}

impl ShellGrid {
    fn surface(
        &mut self,
        world_blocks: &[(Vec3, Material)],
        centroid: Vec3,
        pieces: &[split::Piece],
    ) -> Vec<Grit> {
        self.samples.clear();
        let edge = pieces[0].half.x * 2.0;
        let half = pieces[0].half.x;
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;
        for (center, material) in world_blocks {
            let local_cell = *center - centroid;
            for piece in pieces {
                let local_center = local_cell + piece.local_center;
                let world = *center + piece.local_center;
                let x = (world.x / edge).round() as i32;
                let y = (world.y / edge).round() as i32;
                let z = (world.z / edge).round() as i32;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                min_z = min_z.min(z);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                max_z = max_z.max(z);
                self.samples.push((
                    x,
                    y,
                    z,
                    Grit {
                        center: local_center,
                        half,
                        material: *material,
                    },
                ));
            }
        }
        if self.samples.is_empty() {
            return Vec::new();
        }
        let dx = (max_x - min_x + 1) as usize;
        let dy = (max_y - min_y + 1) as usize;
        let dz = (max_z - min_z + 1) as usize;
        let volume = dx.saturating_mul(dy).saturating_mul(dz);
        if dx == 0 || dy == 0 || dz == 0 || volume > 250_000 {
            return self.surface_hashed();
        }
        self.occupied.clear();
        self.occupied.resize(volume, 0);
        self.centers.resize(volume, Vec3::ZERO);
        self.materials.resize(volume, Material::Concrete);
        for (x, y, z, grit) in &self.samples {
            let ix = (*x - min_x) as usize;
            let iy = (*y - min_y) as usize;
            let iz = (*z - min_z) as usize;
            let i = ix + dx * (iy + dy * iz);
            self.occupied[i] = 1;
            self.centers[i] = grit.center;
            self.materials[i] = grit.material;
        }
        let mut out = Vec::new();
        for iz in 0..dz {
            for iy in 0..dy {
                for ix in 0..dx {
                    let i = ix + dx * (iy + dy * iz);
                    if self.occupied[i] == 0 {
                        continue;
                    }
                    let material = self.materials[i];
                    let enclosed = self.same(ix, iy, iz, 1, 0, 0, dx, dy, dz, material)
                        && self.same(ix, iy, iz, -1, 0, 0, dx, dy, dz, material)
                        && self.same(ix, iy, iz, 0, 1, 0, dx, dy, dz, material)
                        && self.same(ix, iy, iz, 0, -1, 0, dx, dy, dz, material)
                        && self.same(ix, iy, iz, 0, 0, 1, dx, dy, dz, material)
                        && self.same(ix, iy, iz, 0, 0, -1, dx, dy, dz, material);
                    if !enclosed {
                        out.push(Grit {
                            center: self.centers[i],
                            half,
                            material,
                        });
                    }
                }
            }
        }
        out
    }

    fn same(
        &self,
        ix: usize,
        iy: usize,
        iz: usize,
        ox: isize,
        oy: isize,
        oz: isize,
        dx: usize,
        dy: usize,
        dz: usize,
        material: Material,
    ) -> bool {
        let x = ix as isize + ox;
        let y = iy as isize + oy;
        let z = iz as isize + oz;
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= dx || y >= dy || z >= dz {
            return false;
        }
        let j = x + dx * (y + dy * z);
        self.occupied[j] != 0 && self.materials[j] == material
    }

    fn surface_hashed(&mut self) -> Vec<Grit> {
        let mut cubes_by_cell = std::collections::HashMap::with_capacity(self.samples.len());
        for (x, y, z, grit) in &self.samples {
            cubes_by_cell.insert((*x, *y, *z), *grit);
        }
        cubes_by_cell
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
            .collect()
    }
}

/// Greedy boxes for a fallen chunk. More than four boxes becomes one bounds box so a
/// jagged cluster does not hand the solver a collider per block.
fn collider_boxes(blocks: &[CollapsedBlock], block_size: f32) -> Vec<(Vec3, Vec3)> {
    let boxes = compound_boxes(blocks, block_size);
    if boxes.len() <= 4 || blocks.is_empty() {
        return boxes;
    }
    let mut min = blocks[0].local_center;
    let mut max = blocks[0].local_center;
    for block in blocks {
        min = min.min(block.local_center);
        max = max.max(block.local_center);
    }
    let cube = block_size * 0.5;
    let local = (min + max) * 0.5;
    let half = (max - min) * 0.5 + Vec3::new(cube, cube, cube);
    vec![(local, half)]
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
        // local_center is the block centre, and the cell index is that centre divided by
        // the block edge. The box centre is the middle of those centres, not half a block higher.
        let center = Vec3::new(
            (start.0 as f32 + sx as f32 * 0.5 - 0.5) * block_size,
            (start.1 as f32 + sy as f32 * 0.5 - 0.5) * block_size,
            (start.2 as f32 + sz as f32 * 0.5 - 0.5) * block_size,
        );
        let half = Vec3::new(
            sx as f32 * block_size * 0.5,
            sy as f32 * block_size * 0.5,
            sz as f32 * block_size * 0.5,
        );
        out.push((center, half));
    }
    if out.is_empty() {
        out.push((
            Vec3::ZERO,
            Vec3::new(block_size, block_size, block_size) * 0.5,
        ));
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

fn stride_scale(player_height_m: f32) -> f32 {
    player_height_m / STRIDE_REFERENCE_M
}

fn rubble_lifetime(material: Material, lifetime: f32) -> f32 {
    if material.shatters() {
        GLASS_GONE_SECS
    } else {
        lifetime
    }
}

fn rubble_lifetime_of(lifetime: f32, blocks: &[CollapsedBlock]) -> f32 {
    if blocks.iter().any(|block| block.material.shatters()) {
        GLASS_GONE_SECS
    } else {
        lifetime
    }
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

fn fair_shares(n: usize, budget: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![budget / n; n];
    let mut extra = budget % n;
    for share in &mut out {
        if extra == 0 {
            break;
        }
        *share += 1;
        extra -= 1;
    }
    out
}

fn longest_axis(size: Vec3) -> (usize, f32) {
    if size.x >= size.y && size.x >= size.z {
        (0, size.x)
    } else if size.y >= size.z {
        (1, size.y)
    } else {
        (2, size.z)
    }
}

fn cube_bins(share: usize) -> usize {
    let mut n = 1usize;
    let mut k = 2usize;
    while k <= 10 && k * k * k <= share {
        n = k;
        k += 1;
    }
    n
}

fn spatial_groups(mut pieces: Vec<split::Piece>, k: usize) -> Vec<Vec<split::Piece>> {
    if pieces.is_empty() {
        return Vec::new();
    }
    let k = k.max(1).min(pieces.len());
    if k == 1 {
        return vec![pieces];
    }
    let mut min = pieces[0].local_center;
    let mut max = min;
    for piece in &pieces {
        min = min.min(piece.local_center);
        max = max.max(piece.local_center);
    }
    let extent = max - min;
    let n = cube_bins(k);
    if n >= 2 {
        let mut bins = vec![Vec::new(); n * n * n];
        let span = Vec3::new(
            extent.x.max(1.0e-6),
            extent.y.max(1.0e-6),
            extent.z.max(1.0e-6),
        );
        let nf = n as f32;
        for piece in pieces.iter().copied() {
            let ix = (((piece.local_center.x - min.x) / span.x).clamp(0.0, 0.999) * nf) as usize;
            let iy = (((piece.local_center.y - min.y) / span.y).clamp(0.0, 0.999) * nf) as usize;
            let iz = (((piece.local_center.z - min.z) / span.z).clamp(0.0, 0.999) * nf) as usize;
            bins[ix + n * (iy + n * iz)].push(piece);
        }
        bins.retain(|bin| !bin.is_empty());
        if bins.len() >= 2 {
            return bins;
        }
    }
    let (axis, _) = longest_axis(extent);
    pieces.sort_by(|a, b| {
        let coord = |p: &split::Piece| match axis {
            0 => p.local_center.x,
            1 => p.local_center.y,
            _ => p.local_center.z,
        };
        coord(a)
            .partial_cmp(&coord(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut groups = Vec::new();
    for i in 0..k {
        let start = i * pieces.len() / k;
        let end = (i + 1) * pieces.len() / k;
        if start < end {
            groups.push(pieces[start..end].to_vec());
        }
    }
    groups
}

fn sphere_hits_aabb(center: Vec3, radius: f32, box_center: Vec3, half: Vec3) -> bool {
    let closest = center.clamp(box_center - half, box_center + half);
    let d = center - closest;
    d.dot(d) <= radius * radius
}

fn shrunk_span(min: f32, max: f32, shrink: f32) -> (f32, f32) {
    let low = min + shrink;
    let high = max - shrink;
    if low <= high {
        (low, high)
    } else {
        let mid = (min + max) * 0.5;
        (mid, mid)
    }
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
    fn street_load_builds_standing_colliders() {
        let mut cfg = Config::default();
        cfg.block_size = 0.1;
        cfg.level = 2;
        let mut sim = Simulation::new(cfg);
        let started = std::time::Instant::now();
        sim.reload_geometry();
        let ms = started.elapsed().as_millis();
        assert!(sim.block_count() > 50_000);
        assert!(ms < 4_000, "street load took {ms} ms");
        assert!(sim.take_static_meshes().is_some());
        let draw = std::time::Instant::now();
        let mut packed = 0;
        for _ in 0..20 {
            packed = sim.instances().len();
        }
        let draw_us = draw.elapsed().as_micros() / 20;
        assert!(packed > 50_000);
        assert!(
            draw_us < 12_000,
            "packing street instances took {draw_us} us"
        );
    }

    fn horizontal_travel(player_height_m: f32) -> f32 {
        let config = Config {
            player_height_m,
            ..Config::default()
        };
        let mut sim = Simulation::new(config);
        let z0 = sim.player.pos.z;
        sim.input.forward = 1.0;
        sim.move_player(0.5);
        (sim.player.pos.z - z0).abs()
    }

    #[test]
    fn player_height_moves_the_camera_and_scales_stride() {
        let config = Config {
            player_height_m: 1.5,
            ..Config::default()
        };
        let mut sim = Simulation::new(config);
        sim.place_player(Vec3::new(0.0, 2.0, 0.0), 0.0, 0.0);

        assert!((sim.player.pos.y - 2.75).abs() < 1.0e-5);
        assert!((sim.eye().y - 3.38).abs() < 1.0e-5);
        assert!((sim.feet_y() - 2.0).abs() < 1.0e-5);

        sim.config.player_height_m = 0.8;
        assert!(
            (sim.eye().y - 2.68).abs() < 1.0e-5,
            "camera stayed at {:.3}",
            sim.eye().y
        );
        assert!((sim.feet_y() - 2.0).abs() < 1.0e-5);

        sim.config.player_height_m = 1.7;
        sim.sync_player_height();
        assert!((sim.player.pos.y - 2.85).abs() < 1.0e-5);
        assert!((sim.eye().y - 3.58).abs() < 1.0e-5);
        assert!((sim.feet_y() - 2.0).abs() < 1.0e-5);

        let short = horizontal_travel(0.85);
        let tall = horizontal_travel(1.7);
        assert!(
            (tall / short - 2.0).abs() < 0.05,
            "stride ratio {tall:.3} / {short:.3}"
        );
    }

    fn glass_rubble(sim: &Simulation) -> usize {
        let debris = sim
            .debris
            .iter()
            .filter(|debris| debris.material == Material::Glass)
            .count();
        let grit = sim
            .grit_chunks
            .iter()
            .filter(|chunk| {
                chunk
                    .cubes
                    .iter()
                    .any(|cube| cube.material == Material::Glass)
            })
            .count();
        let collapsed = sim
            .collapsed
            .iter()
            .filter(|chunk| {
                chunk
                    .blocks
                    .iter()
                    .any(|block| block.material == Material::Glass)
                    || chunk
                        .cubes
                        .iter()
                        .any(|cube| cube.material == Material::Glass)
            })
            .count();
        debris + grit + collapsed
    }

    fn foliage_rubble(sim: &Simulation) -> usize {
        let want = |material: Material| matches!(material, Material::Trunk | Material::Leaf);
        let debris = sim
            .debris
            .iter()
            .filter(|debris| want(debris.material))
            .count();
        let grit = sim
            .grit_chunks
            .iter()
            .filter(|chunk| chunk.cubes.iter().any(|cube| want(cube.material)))
            .count();
        let collapsed = sim
            .collapsed
            .iter()
            .filter(|chunk| {
                chunk.blocks.iter().any(|block| want(block.material))
                    || chunk.cubes.iter().any(|cube| want(cube.material))
            })
            .count();
        debris + grit + collapsed
    }

    #[test]
    fn broken_glass_is_gone_after_two_seconds() {
        let mut cfg = Config::default();
        cfg.fixed_dt = 1.0 / 120.0;
        cfg.block_size = 0.2;
        cfg.fragment_lifetime_secs = 5.0;
        cfg.adhesive_glass = 0.0;
        let mut sim = Simulation::new(cfg.clone());
        sim.voxels.block_size = cfg.block_size;
        sim.voxels.set(0, 0, 0, Material::Concrete, true);
        sim.voxels.set(0, 1, 0, Material::Concrete, true);
        // A pane and a plaster neighbour, neither touching the floor.
        sim.voxels.set(8, 4, 0, Material::Glass, false);
        sim.voxels.set(9, 4, 0, Material::Plaster, false);
        // This pane is held up, so only the later shot breaks it.
        sim.voxels.set(0, 2, 0, Material::Glass, false);
        sim.refresh_static();

        // Empty air still detaches anything that is not held up.
        sim.explode_at(Vec3::new(-4.0, 6.0, -4.0), 0.15);
        assert!(
            !sim.voxels.blocks.contains_key(&IVec3::new(8, 4, 0)),
            "the loose pane stayed a voxel"
        );
        assert!(
            glass_rubble(&sim) > 0,
            "the loose pane left no glass to see"
        );
        assert!(
            sim.collapsed.iter().any(|chunk| {
                chunk
                    .blocks
                    .iter()
                    .any(|block| block.material == Material::Plaster)
                    && chunk
                        .blocks
                        .iter()
                        .all(|block| block.material != Material::Glass)
            }),
            "plaster rubble still contains the glass"
        );

        let steps = (1.0 / cfg.fixed_dt).round() as i32;
        for _ in 0..steps {
            sim.step(cfg.fixed_dt);
        }
        assert!(glass_rubble(&sim) > 0, "glass vanished before 2 seconds");

        for _ in 0..steps + 12 {
            sim.step(cfg.fixed_dt);
        }
        assert_eq!(glass_rubble(&sim), 0, "glass is still on the ground");
        assert!(
            !sim.voxels.blocks.contains_key(&IVec3::new(8, 4, 0)),
            "the loose pane came back as a voxel"
        );
        assert!(
            sim.voxels
                .blocks
                .get(&IVec3::new(0, 2, 0))
                .is_some_and(|block| block.material == Material::Glass),
            "the supported pane was removed before the shot"
        );
        assert!(
            sim.collapsed.iter().any(|chunk| {
                chunk
                    .blocks
                    .iter()
                    .any(|block| block.material == Material::Plaster)
            }),
            "the plaster pile was removed with the glass"
        );

        // A shot pane becomes shards or a chunk, then that is gone too.
        sim.explode_at(sim.voxels.cell_center(IVec3::new(0, 2, 0)), 0.3);
        assert!(
            !sim.voxels.blocks.contains_key(&IVec3::new(0, 2, 0)),
            "the shot pane stayed a voxel"
        );
        assert!(glass_rubble(&sim) > 0, "the shot produced no glass rubble");
        let later = (2.05 / cfg.fixed_dt).round() as i32;
        for _ in 0..later {
            sim.step(cfg.fixed_dt);
        }
        assert_eq!(glass_rubble(&sim), 0, "shot glass is still on the ground");
    }

    /// The rocket is a ball. Along the ground its skin hits the lawn before the centre
    /// line reaches a low leaf, and that used to leave the tree standing.
    #[test]
    fn low_shot_explodes_a_tree_and_the_pieces_are_gone() {
        let mut cfg = Config::default();
        cfg.block_size = 0.5;
        cfg.explosion_radius = 0.3;
        cfg.blast_impulse = 800.0;
        cfg.fragment_lifetime_secs = 5.0;
        cfg.fixed_dt = 1.0 / 120.0;
        cfg.adhesive_grass = 8.0;
        cfg.adhesive_wood = 22.0;
        let mut sim = Simulation::new(cfg.clone());
        sim.voxels.block_size = cfg.block_size;
        for x in 0..=20 {
            for z in -2..=2 {
                sim.voxels.set(x, 0, z, Material::Grass, true);
            }
        }
        for y in 1..=5 {
            sim.voxels.set(10, y, 0, Material::Trunk, y == 1);
        }
        let low_leaf = IVec3::new(10, 1, 1);
        let crown = IVec3::new(10, 6, 0);
        sim.voxels
            .set(low_leaf.x, low_leaf.y, low_leaf.z, Material::Leaf, false);
        sim.voxels
            .set(crown.x, crown.y, crown.z, Material::Leaf, false);
        let beam = IVec3::new(2, 2, 0);
        sim.voxels.set(beam.x, beam.y, beam.z, Material::Wood, true);
        sim.refresh_static();
        // Colliders join the query tree on the next step. A shot in play happens after that.
        sim.step(cfg.fixed_dt);

        let eye = Vec3::new(0.5, 1.18, 0.75);
        let aim = Vec3::new(5.0, 0.55, 0.75);
        let delta = aim - eye;
        let dist = delta.length();
        let dir = delta * (1.0 / dist);
        let point = sim
            .sweep_ball(eye, dir * 80.0, (dist + 0.5) / 80.0, None)
            .expect("the low leaf is in front of the eye");
        sim.explode_at(point, cfg.explosion_radius);

        assert!(
            !sim.voxels.blocks.contains_key(&low_leaf),
            "a leaf at the ground was not hit, impact {point:?}"
        );
        assert!(
            !sim.voxels.blocks.contains_key(&crown),
            "the crown stayed after the leaf was shot"
        );
        assert!(
            !sim.voxels.blocks.contains_key(&IVec3::new(10, 4, 0)),
            "the trunk stayed in one piece"
        );
        assert!(
            sim.voxels.blocks.contains_key(&beam),
            "a timber beam shattered with the tree"
        );
        assert!(
            sim.voxels.blocks.contains_key(&IVec3::new(2, 0, 0)),
            "the lawn away from the tree was erased"
        );
        assert!(
            foliage_rubble(&sim) > 0,
            "the tree was deleted with nothing to see"
        );

        let steps = (2.15 / cfg.fixed_dt).round() as i32;
        for _ in 0..steps {
            sim.step(cfg.fixed_dt);
        }
        assert_eq!(
            foliage_rubble(&sim),
            0,
            "trunk and leaf pieces are still on the ground"
        );
        assert!(
            sim.voxels.blocks.contains_key(&beam),
            "the timber beam was removed with the tree"
        );

        let lawn = IVec3::new(2, 0, 0);
        sim.explode_at(sim.voxels.cell_center(lawn), 0.3);
        assert!(
            !sim.voxels.blocks.contains_key(&lawn),
            "the lawn cell was not hit"
        );
        assert!(
            sim.debris
                .iter()
                .any(|debris| debris.material == Material::Grass),
            "the lawn left no debris"
        );
        for _ in 0..steps {
            sim.step(cfg.fixed_dt);
        }
        let grass_left = sim
            .debris
            .iter()
            .filter(|debris| debris.material == Material::Grass)
            .count()
            + sim
                .collapsed
                .iter()
                .filter(|chunk| {
                    chunk
                        .blocks
                        .iter()
                        .any(|block| block.material == Material::Grass)
                })
                .count();
        assert!(grass_left > 0, "lawn rubble vanished with the leaves");
    }

    #[test]
    fn hit_wall_piece_falls_when_nothing_is_under_it() {
        let mut cfg = Config::default();
        cfg.block_size = 0.1;
        cfg.explosion_radius = 0.25;
        cfg.blast_impulse = 7400.0;
        cfg.adhesive_stone = 300.0;
        cfg.fixed_dt = 1.0 / 120.0;
        let mut sim = Simulation::new(cfg.clone());
        sim.voxels.block_size = cfg.block_size;
        for y in 0..5 {
            sim.voxels.set(0, y, 0, Material::Stone, true);
        }
        for x in 1..5 {
            sim.voxels.set(x, 3, 0, Material::Stone, false);
        }
        sim.refresh_static();
        let at = sim.voxels.cell_center(IVec3::new(2, 3, 0));
        let y0 = at.y;
        sim.explode_at(at, cfg.explosion_radius);
        assert!(
            sim.grit_count() + sim.collapsed_count() + sim.debris_count() > 0,
            "the hit produced no rigid pieces"
        );
        let steps = (0.5 / cfg.fixed_dt) as i32;
        for _ in 0..steps {
            sim.step(cfg.fixed_dt);
        }
        let mut lowest = y0;
        if let Some(y) = sim.grit_lowest_y() {
            lowest = lowest.min(y);
        }
        if let Some(y) = sim.highest_collapsed_y() {
            lowest = lowest.min(y);
        }
        for debris in &sim.debris {
            if let Some(pos) = sim.body_translation(debris.body) {
                lowest = lowest.min(pos.y - debris.half.y);
            }
        }
        assert!(
            y0 - lowest > 0.4,
            "hit pieces stayed up: start {y0:.3} lowest {lowest:.3}"
        );
    }

    #[test]
    fn load_level_switches_villa_and_biome() {
        let mut sim = Simulation::new(Config::default());
        sim.config.level = 0;
        sim.reload_geometry();
        assert!(sim.ground_height(10.0, 10.0).is_none());
        assert!(sim
            .voxels
            .blocks
            .values()
            .any(|block| block.material == Material::Glass));
        sim.config.level = 1;
        sim.reload_geometry();
        assert!(sim.ground_height(10.0, 10.0).is_some());
        assert!(sim
            .voxels
            .blocks
            .values()
            .all(|block| block.material != Material::Glass));
        sim.config.level = 0;
        sim.reload_geometry();
        assert!(sim.ground_height(10.0, 10.0).is_none());
        assert!(sim
            .voxels
            .blocks
            .values()
            .any(|block| block.material == Material::Glass));
    }

    #[test]
    fn biome_blast_throws_a_tree_and_the_hill_stays() {
        let mut sim = Simulation::new(Config::default());
        sim.config.level = 1;
        sim.reload_geometry();
        let before = sim.block_count();
        let terrain_before = sim.terrain_collider_handle();
        assert!(terrain_before.is_some(), "the biome has no heightfield");
        let hill = sim.ground_height(800.0, 800.0).expect("hill");
        let tree = sim
            .voxels
            .blocks
            .iter()
            .find(|(_, block)| block.material == Material::Trunk)
            .map(|(cell, _)| *cell)
            .expect("a tree");
        let at = sim.voxels.cell_center(tree);
        sim.explode_at(at, sim.config.explosion_radius);
        assert_eq!(
            sim.terrain_collider_handle(),
            terrain_before,
            "a tree shot rebuilt the biome heightfield"
        );
        assert!(
            sim.block_count() < before,
            "the tree was not removed from the grid"
        );
        assert!(
            sim.debris_count() + sim.collapsed_count() + sim.grit_count() > 0,
            "the blast erased the tree instead of throwing it"
        );
        let hill_after = sim.ground_height(800.0, 800.0).expect("hill still there");
        assert!((hill - hill_after).abs() < 1.0e-4);
        for _ in 0..30 {
            sim.step(sim.config.fixed_dt);
        }
        let feet = sim.feet_y();
        let under = sim
            .ground_height(sim.player.pos.x, sim.player.pos.z)
            .expect("ground under the player");
        assert!(
            (feet - under).abs() < 2.5,
            "player fell through the hill: feet {feet:.2} ground {under:.2}"
        );
    }

    /// A coarse step used to move an unsupported body on the pure ballistic curve, through a
    /// thick floor. The body then overlapped the street for the rest of the fall.
    #[test]
    fn coarse_dt_keeps_a_thrown_block_on_a_thick_floor() {
        let dt = 1.0 / 30.0;
        let mut cfg = Config::default();
        cfg.fixed_dt = dt;
        cfg.block_size = 0.5;
        let mut sim = Simulation::new(cfg);
        for x in -10..=10 {
            for z in -10..=10 {
                sim.voxels.set(x, 0, z, Material::Concrete, true);
                sim.voxels.set(x, 1, z, Material::Concrete, true);
            }
        }
        sim.refresh_static();
        let floor_top = 2.0 * sim.voxels.block_size;
        let half = Vec3::new(0.2, 0.2, 0.2);
        let probe = sim.spawn_probe(Vec3::new(0.0, floor_top + half.y + 0.02, 0.0), half);
        sim.physics
            .bodies
            .get_mut(probe)
            .unwrap()
            .set_linvel(Vector::new(1.5, -40.0, 0.0), true);
        let steps = (1.5 / dt).round() as i32;
        let mut lowest = f32::MAX;
        for _ in 0..steps {
            sim.step(dt);
            let y = sim
                .body_translation(probe)
                .expect("the block fell out of the world")
                .y;
            lowest = lowest.min(y);
        }
        let bottom = lowest - half.y;
        assert!(
            bottom >= floor_top - 1.0e-3,
            "thrown block crossed the floor: lowest bottom {bottom:.4} m, floor {floor_top:.4} m"
        );
        let y = sim.body_translation(probe).unwrap().y;
        assert!(
            y < floor_top + 1.0,
            "thrown block left the floor: center y {y:.4} m"
        );
        let (bodies, colliders, _) = sim.physics_stats();
        assert_eq!(bodies, 2, "floor clip spawned bodies ({bodies})");
        assert!(colliders >= 2, "floor lost its collider");
    }

    /// One blast at the coarse timestep must not keep creating bodies, colliders, or contact pairs.
    #[test]
    fn coarse_dt_does_not_grow_the_scene_after_one_blast() {
        let dt = 1.0 / 30.0;
        let mut cfg = Config::default();
        cfg.fixed_dt = dt;
        cfg.fragment_lifetime_secs = 5.0;
        cfg.explosion_radius = 1.5;
        cfg.block_size = 0.5;
        let mut sim = Simulation::new(cfg);
        for x in 0..16 {
            for z in 0..16 {
                sim.voxels.set(x, 0, z, Material::Concrete, true);
                sim.voxels.set(x, 1, z, Material::Concrete, false);
                sim.voxels.set(x, 2, z, Material::Plaster, false);
            }
        }
        sim.refresh_static();
        sim.explode_at(Vec3::new(2.0, 0.4, 2.0), 1.5);
        while sim.time < 1.5 {
            sim.step(dt);
        }
        let (bodies_0, colliders_0, _) = sim.physics_stats();
        let pairs_0 = sim.physics.narrow_phase.contact_pairs().count();
        let blocks_0 = sim.block_count();
        let mut max_bodies = bodies_0;
        let mut max_colliders = colliders_0;
        let mut max_pairs = pairs_0;
        while sim.time < 4.0 {
            sim.step(dt);
            let (bodies, colliders, _) = sim.physics_stats();
            let pairs = sim.physics.narrow_phase.contact_pairs().count();
            max_bodies = max_bodies.max(bodies);
            max_colliders = max_colliders.max(colliders);
            max_pairs = max_pairs.max(pairs);
            assert!(
                sim.block_count() <= blocks_0,
                "voxels kept breaking after the blast: {blocks_0} then {}",
                sim.block_count()
            );
        }
        assert!(
            max_bodies <= bodies_0,
            "bodies grew after the blast settled: {bodies_0} then {max_bodies}"
        );
        assert!(
            max_colliders <= colliders_0,
            "colliders grew after the blast settled: {colliders_0} then {max_colliders}"
        );
        assert!(
            max_pairs <= pairs_0.saturating_add(16),
            "contact pairs grew after the blast settled: {pairs_0} then {max_pairs}"
        );
        assert!(
            sim.static_colliders_match_voxels(),
            "static colliders no longer match the voxels"
        );
    }

    /// A collapsed wall can be lying on its side. A later shot at where it is drawn has to
    /// cut that block up, and a piece that is still larger than the minimum cube has to cut again.
    #[test]
    fn fallen_construction_breaks_smaller_after_it_has_turned() {
        let mut cfg = Config::default();
        cfg.block_size = 0.5;
        cfg.fragment_count = 1000;
        cfg.min_fragment_extent = 0.005;
        cfg.fixed_dt = 1.0 / 120.0;
        let mut sim = Simulation::new(cfg);
        for x in -2..=2 {
            for z in -2..=2 {
                sim.voxels.set(x, 0, z, Material::Concrete, true);
            }
        }
        sim.voxels.set(0, 1, 0, Material::Stone, false);
        sim.voxels.set(0, 2, 0, Material::Stone, false);
        sim.voxels.set(0, 3, 0, Material::Stone, false);
        sim.refresh_static();
        sim.erase_cells(&[IVec3::new(0, 0, 0)]);
        assert_eq!(
            sim.collapsed_count(),
            1,
            "the column did not fall as one piece"
        );
        assert_eq!(sim.collapsed[0].blocks.len(), 3);

        let origin = sim
            .body_translation(sim.collapsed[0].body)
            .expect("collapsed body");
        let locals: Vec<Vec3> = sim.collapsed[0]
            .blocks
            .iter()
            .map(|block| block.local_center)
            .collect();
        let handle = sim.collapsed[0].body;
        sim.physics.bodies.get_mut(handle).unwrap().set_rotation(
            Rotation::from_axis_angle(Vector::Z, std::f32::consts::FRAC_PI_2),
            true,
        );
        // +90° about Z sends a vertical column onto its side: (x, y) -> (-y, x).
        let visual: Vec<Vec3> = locals
            .iter()
            .map(|local| origin + Vec3::new(-local.y, local.x, local.z))
            .collect();
        let y0 = visual[0].y;
        assert!(
            visual.iter().all(|p| (p.y - y0).abs() < 1.0e-4),
            "the column is still standing in the hit test"
        );
        let (hit_i, _) = visual
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.x.partial_cmp(&b.x).unwrap())
            .unwrap();
        let hit = visual[hit_i];
        let kept: Vec<Vec3> = visual
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| *i != hit_i)
            .map(|(_, p)| p)
            .collect();

        let s = sim.voxels.block_size;
        sim.explode_at(hit, s * 0.2);

        let after = collapsed_block_worlds(&sim);
        assert_eq!(
            after.len(),
            2,
            "the unhit blocks were not left as construction"
        );
        assert!(
            near(&after, kept[0]) && near(&after, kept[1]),
            "unhit blocks left their rotated places: {after:?} kept {kept:?}"
        );
        assert!(
            !near(&after, hit),
            "the block under the blast is still a full construction block"
        );
        assert!(
            has_piece_smaller_than(&sim, s * 0.5),
            "the hit block was not cut into smaller blocks"
        );
        assert!(
            sim.debris_count() >= 8 && sim.debris_count() <= RUBBLE_BREAK_SLOTS,
            "one block spawned {} debris",
            sim.debris_count()
        );
        assert_eq!(sim.grit_count(), 0, "part of the hit block stayed fused");
        let volume: f32 = sim
            .debris
            .iter()
            .map(|debris| debris.full_size.x * debris.full_size.y * debris.full_size.z)
            .sum();
        assert!(
            (volume - s * s * s).abs() < 1.0e-3,
            "the pieces do not fill the hit block: volume {volume} block {}",
            s * s * s
        );
        assert!(
            sim.debris.iter().all(|debris| {
                debris
                    .full_size
                    .x
                    .max(debris.full_size.y)
                    .max(debris.full_size.z)
                    < s * 0.5
            }),
            "a piece is still most of the block"
        );

        let candidates: Vec<(RigidBodyHandle, f32)> = sim
            .debris
            .iter()
            .map(|debris| {
                let edge = debris
                    .full_size
                    .x
                    .max(debris.full_size.y)
                    .max(debris.full_size.z);
                (debris.body, edge)
            })
            .collect();
        let target = candidates
            .iter()
            .filter_map(|(handle, edge)| {
                let pos = sim.body_translation(*handle)?;
                let clear = after
                    .iter()
                    .map(|block| (pos - *block).length())
                    .fold(f32::MAX, f32::min);
                Some((clear, *handle, pos, *edge))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .expect("no block/10 fragment to shoot again");
        assert!(
            target.0 > s * 0.4,
            "every fragment is still inside the remaining construction"
        );
        let handle = target.1;
        let at = target.2;
        sim.explode_at(at, s * 0.02);
        assert!(
            sim.physics.bodies.get(handle).is_none(),
            "the fragment that was shot is still the same body"
        );
        assert!(
            has_piece_smaller_than(&sim, s / 20.0),
            "a fragment that was already hit did not break into smaller blocks"
        );
        assert!(
            sim.debris_count() <= RUBBLE_BREAK_SLOTS * 2,
            "the second shot spawned {} debris",
            sim.debris_count()
        );
    }

    fn collapsed_block_worlds(sim: &Simulation) -> Vec<Vec3> {
        let mut worlds = Vec::new();
        for chunk in &sim.collapsed {
            let Some(body) = sim.physics.bodies.get(chunk.body) else {
                continue;
            };
            let origin = v_from(body.translation());
            let rotation = quat_of(body.rotation());
            for block in &chunk.blocks {
                worlds.push(origin + crate::math3::quat_rotate(rotation, block.local_center));
            }
        }
        worlds
    }

    fn near(worlds: &[Vec3], p: Vec3) -> bool {
        worlds.iter().any(|w| (*w - p).length() < 1.0e-3)
    }

    fn has_piece_smaller_than(sim: &Simulation, edge: f32) -> bool {
        sim.debris.iter().any(|debris| {
            debris
                .full_size
                .x
                .max(debris.full_size.y)
                .max(debris.full_size.z)
                < edge
        }) || sim
            .grit_chunks
            .iter()
            .any(|chunk| chunk.cubes.iter().any(|cube| cube.half * 2.0 < edge))
    }
}
