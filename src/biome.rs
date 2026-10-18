//! One square kilometre of hills, plus the things a blast can throw.
//!
//! The ground is a heightfield, not a voxel slab. A 0.215 m grid over 1 km² is
//! about twenty million cells for a single layer, and the hills would multiply
//! that. Those cells are not destructible, so they are not blocks. Trees, bushes,
//! and rocks are blocks. Only the cell that touches the ground is anchored, so a
//! later blast can sever a trunk and the crown falls. Anchoring every cell would
//! leave the crown hanging. Anchoring none would drop every tree on the first shot
//! anywhere on the map, because nothing here is connected to an anchored voxel floor.

use crate::material::Material;
use crate::math3::{IVec3, Vec3};
use crate::mesh::{Mesh, Vertex};
use crate::villa::Spawn;
use crate::world::VoxelWorld;

pub const WIDTH: f32 = 1000.0;
const SAMPLES: usize = 251;
const BLOCK_BUDGET: usize = 40_000;

pub struct Terrain {
    pub width: f32,
    pub samples: usize,
    /// Column-major heights. Index `z + x * samples` matches parry's `Array2`.
    heights: Vec<f32>,
    pub mesh: Mesh,
}

impl Terrain {
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let n = (self.samples - 1) as f32;
        let u = (x / self.width).clamp(0.0, 1.0) * n;
        let v = (z / self.width).clamp(0.0, 1.0) * n;
        let x0 = (u as usize).min(self.samples - 2);
        let z0 = (v as usize).min(self.samples - 2);
        let tx = u - x0 as f32;
        let tz = v - z0 as f32;
        let h00 = self.sample(z0, x0);
        let h10 = self.sample(z0, x0 + 1);
        let h01 = self.sample(z0 + 1, x0);
        let h11 = self.sample(z0 + 1, x0 + 1);
        let hx0 = h00 + (h10 - h00) * tx;
        let hx1 = h01 + (h11 - h01) * tx;
        hx0 + (hx1 - hx0) * tz
    }

    pub fn collider_heights(&self) -> Vec<f32> {
        self.heights.clone()
    }

    fn sample(&self, z: usize, x: usize) -> f32 {
        self.heights[z + x * self.samples]
    }
}

pub fn build(world: &mut VoxelWorld) -> (Spawn, Terrain) {
    let terrain = Terrain::generate();
    let spawn = spawn_on(&terrain);
    place_props(world, &terrain, spawn.feet);
    (spawn, terrain)
}

impl Terrain {
    fn generate() -> Self {
        let mut heights = vec![0.0; SAMPLES * SAMPLES];
        for x in 0..SAMPLES {
            for z in 0..SAMPLES {
                let n = (SAMPLES - 1) as f32;
                let wx = x as f32 / n * WIDTH;
                let wz = z as f32 / n * WIDTH;
                heights[z + x * SAMPLES] = elevation(wx, wz);
            }
        }
        let mesh = terrain_mesh(&heights);
        Self {
            width: WIDTH,
            samples: SAMPLES,
            heights,
            mesh,
        }
    }
}

fn elevation(x: f32, z: f32) -> f32 {
    let broad = fbm(x * 0.0032, z * 0.0032, 4);
    let ridge = 1.0 - (fbm(x * 0.006 + 40.0, z * 0.006, 3) * 2.0 - 1.0).abs();
    let detail = fbm(x * 0.018, z * 0.018, 2);
    4.0 + broad * 32.0 + ridge * ridge * 14.0 + detail * 2.0
}

fn fbm(x: f32, z: f32, octaves: i32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += value_noise(x * freq, z * freq) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

fn value_noise(x: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = x - x0 as f32;
    let fz = z - z0 as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let a = hash01(x0, z0);
    let b = hash01(x0 + 1, z0);
    let c = hash01(x0, z0 + 1);
    let d = hash01(x0 + 1, z0 + 1);
    let ab = a + (b - a) * sx;
    let cd = c + (d - c) * sx;
    ab + (cd - ab) * sz
}

fn hash01(x: i32, z: i32) -> f32 {
    let mut n = x.wrapping_mul(374761393) ^ z.wrapping_mul(668265263);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n as u32 & 0x7fff_ffff) as f32 / 0x7fff_ffff as f32
}

fn terrain_mesh(heights: &[f32]) -> Mesh {
    let n = SAMPLES;
    let mut vertices = Vec::with_capacity(n * n);
    for x in 0..n {
        for z in 0..n {
            let wx = x as f32 / (n - 1) as f32 * WIDTH;
            let wz = z as f32 / (n - 1) as f32 * WIDTH;
            let y = heights[z + x * n];
            let slope = slope_at(heights, x, z);
            let normal = normal_at(heights, x, z);
            vertices.push(Vertex {
                position: [wx, y, wz],
                normal,
                color: ground_color(slope, y),
            });
        }
    }
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for x in 0..n - 1 {
        for z in 0..n - 1 {
            let i = (z + x * n) as u32;
            let row = n as u32;
            indices.extend([i, i + 1, i + row, i + 1, i + row + 1, i + row]);
        }
    }
    Mesh { vertices, indices }
}

fn slope_at(heights: &[f32], x: usize, z: usize) -> f32 {
    let n = SAMPLES;
    let step = WIDTH / (n - 1) as f32;
    let x0 = x.saturating_sub(1);
    let x1 = (x + 1).min(n - 1);
    let z0 = z.saturating_sub(1);
    let z1 = (z + 1).min(n - 1);
    let dx = (heights[z + x1 * n] - heights[z + x0 * n]) / ((x1 - x0) as f32 * step);
    let dz = (heights[z1 + x * n] - heights[z0 + x * n]) / ((z1 - z0) as f32 * step);
    (dx * dx + dz * dz).sqrt()
}

fn normal_at(heights: &[f32], x: usize, z: usize) -> [f32; 3] {
    let n = SAMPLES;
    let step = WIDTH / (n - 1) as f32;
    let x0 = x.saturating_sub(1);
    let x1 = (x + 1).min(n - 1);
    let z0 = z.saturating_sub(1);
    let z1 = (z + 1).min(n - 1);
    let dx = (heights[z + x1 * n] - heights[z + x0 * n]) / ((x1 - x0).max(1) as f32 * step);
    let dz = (heights[z1 + x * n] - heights[z0 + x * n]) / ((z1 - z0).max(1) as f32 * step);
    let normal = Vec3::new(-dx, 1.0, -dz).normalized();
    [normal.x, normal.y, normal.z]
}

fn ground_color(slope: f32, height: f32) -> [f32; 4] {
    let grass = [0.32, 0.48, 0.22, 1.0];
    let soil = [0.42, 0.31, 0.18, 1.0];
    let stone = [0.62, 0.60, 0.56, 1.0];
    let t = ((slope - 0.18) / 0.55).clamp(0.0, 1.0);
    let mut color = if t < 0.45 {
        lerp_color(grass, soil, t / 0.45)
    } else {
        lerp_color(soil, stone, (t - 0.45) / 0.55)
    };
    let shade = 0.82 + (height / 50.0).clamp(0.0, 0.18);
    color[0] *= shade;
    color[1] *= shade;
    color[2] *= shade;
    color
}

fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        1.0,
    ]
}

fn spawn_on(terrain: &Terrain) -> Spawn {
    let mut best = (500.0_f32, 500.0_f32, f32::MAX);
    let mut x = 420.0;
    while x <= 580.0 {
        let mut z = 420.0;
        while z <= 580.0 {
            let slope = terrain_slope(terrain, x, z);
            if slope < best.2 {
                best = (x, z, slope);
            }
            z += 8.0;
        }
        x += 8.0;
    }
    let y = terrain.height_at(best.0, best.1);
    Spawn {
        feet: Vec3::new(best.0, y + 0.15, best.1),
        yaw: 0.4,
        pitch: -0.12,
    }
}

fn terrain_slope(terrain: &Terrain, x: f32, z: f32) -> f32 {
    let d = 2.0;
    let dx = terrain.height_at(x + d, z) - terrain.height_at(x - d, z);
    let dz = terrain.height_at(x, z + d) - terrain.height_at(x, z - d);
    (dx * dx + dz * dz).sqrt() / (d * 2.0)
}

#[derive(Clone, Copy)]
enum Prop {
    Small,
    Medium,
    Large,
    Bush,
    Rock,
}

fn place_props(world: &mut VoxelWorld, terrain: &Terrain, spawn: Vec3) {
    // The previous scatter was one object every 18 m, with trunks of about 2 m.
    // On a kilometre of hills that is not a forest: the loaded run had 9326 blocks
    // and the view from the spawn was empty grass. Plant a stand the player can
    // see without hunting, then thin it out.
    let yaw = 0.4_f32;
    let forward = Vec3::new(yaw.sin(), 0.0, -yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);
    let mut rng = Rng(0xB10E_5EED);
    let mut planted = 0u32;
    let mut radius = 7.0_f32;
    while radius < 70.0 && world.blocks.len() + 2_000 < BLOCK_BUDGET {
        let step = 5.2_f32;
        let count = ((std::f32::consts::TAU * radius) / step).max(6.0) as i32;
        for i in 0..count {
            if world.blocks.len() + 2_000 >= BLOCK_BUDGET {
                break;
            }
            let ang = i as f32 / count as f32 * std::f32::consts::TAU;
            let hx = spawn.x + forward.x * radius * ang.cos() + right.x * radius * ang.sin();
            let hz = spawn.z + forward.z * radius * ang.cos() + right.z * radius * ang.sin();
            if hx < 4.0 || hz < 4.0 || hx > WIDTH - 4.0 || hz > WIDTH - 4.0 {
                continue;
            }
            let prop = forest_prop(planted, &mut rng);
            plant(world, terrain, hx, hz, prop, &mut rng);
            planted += 1;
        }
        radius += if radius < 28.0 { 4.6 } else { 6.5 };
    }
    let mut dist = 80.0_f32;
    while dist < 220.0 && world.blocks.len() + 2_000 < BLOCK_BUDGET {
        let hx = (spawn.x + forward.x * dist + right.x * (rng.unit() - 0.5) * 40.0)
            .clamp(8.0, WIDTH - 8.0);
        let hz = (spawn.z + forward.z * dist + right.z * (rng.unit() - 0.5) * 40.0)
            .clamp(8.0, WIDTH - 8.0);
        plant(
            world,
            terrain,
            hx,
            hz,
            forest_prop(planted, &mut rng),
            &mut rng,
        );
        planted += 1;
        dist += 14.0;
    }
}

fn forest_prop(index: u32, rng: &mut Rng) -> Prop {
    match (index + (rng.next_u32() % 3)) % 10 {
        0 | 1 => Prop::Large,
        2 | 3 | 4 => Prop::Medium,
        5 | 6 => Prop::Small,
        7 | 8 => Prop::Bush,
        _ => Prop::Rock,
    }
}

fn plant(world: &mut VoxelWorld, terrain: &Terrain, x: f32, z: f32, prop: Prop, rng: &mut Rng) {
    let s = world.block_size;
    let ground = terrain.height_at(x, z);
    let cx = (x / s).floor() as i32;
    let cz = (z / s).floor() as i32;
    let cy = (ground / s).floor() as i32;
    if world.blocks.contains_key(&IVec3::new(cx, cy, cz)) {
        return;
    }
    match prop {
        Prop::Small => tree(world, cx, cy, cz, 1, 16 + (rng.next_u32() % 6) as i32, 2, 6),
        Prop::Medium => tree(world, cx, cy, cz, 2, 28 + (rng.next_u32() % 8) as i32, 3, 9),
        Prop::Large => tree(
            world,
            cx,
            cy,
            cz,
            2,
            40 + (rng.next_u32() % 10) as i32,
            5,
            12,
        ),
        Prop::Bush => bush(world, cx, cy, cz),
        Prop::Rock => rock(world, cx, cy, cz, 2 + (rng.next_u32() % 3) as i32, rng),
    }
}

fn tree(
    world: &mut VoxelWorld,
    x: i32,
    y: i32,
    z: i32,
    width: i32,
    trunk: i32,
    radius: i32,
    canopy: i32,
) {
    let wood = Material::Trunk;
    let ox = x - width / 2;
    let oz = z - width / 2;
    for dy in 0..trunk {
        for dx in 0..width {
            for dz in 0..width {
                world.set(ox + dx, y + dy, oz + dz, wood, dy == 0);
            }
        }
    }
    let top = y + trunk - 2;
    let cx = x;
    let cz = z;
    for dx in -radius..=radius {
        for dy in 0..canopy {
            for dz in -radius..=radius {
                let nx = dx as f32 / radius.max(1) as f32;
                let ny = dy as f32 / canopy.max(1) as f32 * 2.0 - 0.35;
                let nz = dz as f32 / radius.max(1) as f32;
                if nx * nx + ny * ny + nz * nz > 1.02 {
                    continue;
                }
                let cell = IVec3::new(cx + dx, top + dy, cz + dz);
                if world.blocks.contains_key(&cell) {
                    continue;
                }
                world.set(cell.x, cell.y, cell.z, Material::Leaf, false);
            }
        }
    }
}

fn bush(world: &mut VoxelWorld, x: i32, y: i32, z: i32) {
    for dx in -1..=1 {
        for dz in -1..=1 {
            world.set(x + dx, y, z + dz, Material::Leaf, true);
            if dx != 0 || dz != 0 {
                world.set(x + dx, y + 1, z + dz, Material::Leaf, false);
            }
        }
    }
    world.set(x, y + 1, z, Material::Leaf, false);
    world.set(x, y + 2, z, Material::Trunk, false);
}

fn rock(world: &mut VoxelWorld, x: i32, y: i32, z: i32, size: i32, rng: &mut Rng) {
    let size = size.max(3);
    let h = 2 + size / 3;
    for dx in 0..size {
        for dy in 0..h {
            for dz in 0..size {
                let edge = dx == 0 || dz == 0 || dx == size - 1 || dz == size - 1;
                if edge && dy == h - 1 && rng.unit() > 0.55 {
                    continue;
                }
                world.set(x + dx, y + dy, z + dz, Material::Stone, dy == 0);
            }
        }
    }
}

struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hills_cover_a_square_kilometre_and_are_not_blocks() {
        let mut world = VoxelWorld::new(0.215443);
        let (spawn, terrain) = build(&mut world);
        assert!((terrain.width - 1000.0).abs() < 1.0e-3);
        let mut low = f32::MAX;
        let mut high = f32::MIN;
        for height in &terrain.heights {
            low = low.min(*height);
            high = high.max(*height);
        }
        assert!(
            high - low > 12.0,
            "hills must rise more than a kerb, range {low:.2}..{high:.2}"
        );
        assert!(spawn.feet.x > 0.0 && spawn.feet.x < WIDTH);
        assert!(spawn.feet.z > 0.0 && spawn.feet.z < WIDTH);
        let ground = terrain.height_at(spawn.feet.x, spawn.feet.z);
        assert!((spawn.feet.y - ground - 0.15).abs() < 0.05);
        let mut empty = 0;
        for i in 0..100 {
            let x = 30.0 + (i % 10) as f32 * 90.0;
            let z = 30.0 + (i / 10) as f32 * 90.0;
            let s = world.block_size;
            let cell = IVec3::new(
                (x / s).floor() as i32,
                (terrain.height_at(x, z) / s).floor() as i32,
                (z / s).floor() as i32,
            );
            if !world.blocks.contains_key(&cell) {
                empty += 1;
            }
        }
        assert!(
            empty >= 70,
            "the hill surface is a heightfield; only props occupy cells ({empty}/100 empty)"
        );
        assert!(world.blocks.values().all(|block| !block.indestructible));
    }

    #[test]
    fn props_are_several_tree_sizes_bushes_and_rocks() {
        let mut world = VoxelWorld::new(0.215443);
        let _ = build(&mut world);
        let mut wood = 0;
        let mut grass = 0;
        let mut stone = 0;
        let mut trunk_heights = Vec::new();
        for (cell, block) in &world.blocks {
            match block.material {
                Material::Trunk => wood += 1,
                Material::Leaf => grass += 1,
                Material::Stone => stone += 1,
                _ => {}
            }
            if block.material == Material::Trunk && block.anchored {
                let mut height = 0;
                let mut y = cell.y;
                while world
                    .blocks
                    .get(&IVec3::new(cell.x, y, cell.z))
                    .is_some_and(|above| above.material == Material::Trunk)
                {
                    height += 1;
                    y += 1;
                }
                trunk_heights.push(height);
            }
        }
        assert!(wood > 40, "wood {wood}");
        assert!(grass > 40, "grass {grass}");
        assert!(stone > 20, "stone {stone}");
        let min = *trunk_heights.iter().min().unwrap_or(&0);
        let max = *trunk_heights.iter().max().unwrap_or(&0);
        assert!(
            max - min >= 6,
            "tree trunks must differ in size, range {min}..{max}"
        );
        assert!(
            world.blocks.len() <= BLOCK_BUDGET,
            "blocks {} exceed budget {}",
            world.blocks.len(),
            BLOCK_BUDGET
        );
    }

    #[test]
    fn a_forest_stands_within_sight_of_the_spawn() {
        let mut world = VoxelWorld::new(0.215443);
        let (spawn, _) = build(&mut world);
        let s = world.block_size;
        let mut trees = 0;
        let mut tall = 0;
        let mut bushes = 0;
        let mut rocks = 0;
        let mut seen_wood = std::collections::HashSet::new();
        for (cell, block) in &world.blocks {
            let wx = cell.x as f32 * s;
            let wz = cell.z as f32 * s;
            if (wx - spawn.feet.x).hypot(wz - spawn.feet.z) > 40.0 {
                continue;
            }
            if matches!(block.material, Material::Stone) && block.anchored {
                rocks += 1;
            }
            if block.material == Material::Leaf && block.anchored {
                bushes += 1;
            }
            if block.material == Material::Trunk && block.anchored {
                let key = (cell.x, cell.z);
                if !seen_wood.insert(key) {
                    continue;
                }
                let mut height = 0;
                let mut y = cell.y;
                while world
                    .blocks
                    .get(&IVec3::new(cell.x, y, cell.z))
                    .is_some_and(|above| above.material == Material::Trunk)
                {
                    height += 1;
                    y += 1;
                }
                trees += 1;
                if height as f32 * s > 4.0 {
                    tall += 1;
                }
            }
        }
        assert!(trees >= 12, "trees within 40 m: {trees}");
        assert!(tall >= 4, "trees taller than 4 m: {tall}");
        assert!(bushes >= 4, "bush bases within 40 m: {bushes}");
        assert!(rocks >= 2, "rock bases within 40 m: {rocks}");
    }
}
