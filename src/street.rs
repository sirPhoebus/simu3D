//! A dead-end street in metres. Ten cottages face a square. Trees and cars line it.
//! Cell counts follow the world block size, so a door stays about 2.2 m and a wall about 3 m
//! whether a block is 0.1 m or 0.2 m. The player height setting moves the camera. It does not resize the street.

use crate::material::Material;
use crate::math3::{IVec3, Vec3};
use crate::villa::Spawn;
use crate::world::VoxelWorld;

const HOUSE_W_M: f32 = 6.0;
const HOUSE_D_M: f32 = 4.6;
const WALL_M: f32 = 3.0;
const DOOR_W_M: f32 = 1.0;
const DOOR_H_M: f32 = 2.2;
const CAR_W_M: f32 = 1.75;
const CAR_L_M: f32 = 4.4;
const CAR_H_M: f32 = 1.5;
const TREE_H_M: f32 = 5.5;

const SPAWN_X: f32 = 16.3;
const SPAWN_Z: f32 = 36.4;

#[derive(Clone, Copy)]
enum Face {
    South,
    North,
    East,
    West,
}

#[derive(Clone, Copy)]
struct Villa {
    x: f32,
    z: f32,
    face: Face,
}

#[derive(Clone, Copy)]
struct Car {
    x: f32,
    z: f32,
    body: Material,
}

struct Grid {
    ox: i32,
    oz: i32,
    w: i32,
    d: i32,
}

struct Shell {
    w: i32,
    d: i32,
    door_h: i32,
    door_x0: i32,
    door_x1: i32,
    wall_top: i32,
    win_y0: i32,
    win_y1: i32,
    mullion: i32,
}

pub fn build(world: &mut VoxelWorld) -> Spawn {
    if world.block_size <= 0.12 {
        world.blocks.reserve(280_000);
    }
    ground(world);
    sidewalks(world);
    road(world);
    let block = world.block_size;
    for villa in villas() {
        house(world, villa, block);
    }
    for (x, z) in trees() {
        tree(world, x, z);
    }
    for car in cars() {
        park(world, car);
    }

    Spawn {
        feet: Vec3::new(SPAWN_X, block, SPAWN_Z),
        yaw: 0.0,
        pitch: -0.06,
    }
}

fn villas() -> [Villa; 10] {
    // North row faces south into the square. Side rows face inward.
    // The south pair flanks the road, so the street is the only way in.
    [
        Villa {
            x: 1.0,
            z: 1.0,
            face: Face::South,
        },
        Villa {
            x: 9.2,
            z: 1.0,
            face: Face::South,
        },
        Villa {
            x: 17.4,
            z: 1.0,
            face: Face::South,
        },
        Villa {
            x: 25.6,
            z: 1.0,
            face: Face::South,
        },
        Villa {
            x: 1.0,
            z: 7.0,
            face: Face::East,
        },
        Villa {
            x: 1.0,
            z: 15.2,
            face: Face::East,
        },
        Villa {
            x: 27.0,
            z: 7.0,
            face: Face::West,
        },
        Villa {
            x: 27.0,
            z: 15.2,
            face: Face::West,
        },
        Villa {
            x: 6.3,
            z: 22.8,
            face: Face::North,
        },
        Villa {
            x: 20.3,
            z: 22.8,
            face: Face::North,
        },
    ]
}

fn trees() -> [(f32, f32); 14] {
    [
        (13.0, 8.4),
        (16.3, 8.0),
        (19.6, 8.4),
        (13.0, 14.6),
        (16.3, 12.4),
        (19.6, 14.6),
        (16.3, 17.0),
        (10.4, 10.8),
        (24.4, 12.8),
        (9.2, 31.2),
        (9.2, 35.8),
        (23.4, 31.2),
        (23.4, 35.8),
        (11.6, 17.8),
    ]
}

fn cars() -> [Car; 8] {
    [
        Car {
            x: 7.4,
            z: 7.6,
            body: Material::Metal,
        },
        Car {
            x: 7.4,
            z: 13.2,
            body: Material::Plaster,
        },
        Car {
            x: 22.2,
            z: 7.6,
            body: Material::Fabric,
        },
        Car {
            x: 22.2,
            z: 13.2,
            body: Material::Metal,
        },
        Car {
            x: 11.2,
            z: 29.0,
            body: Material::Plaster,
        },
        Car {
            x: 11.2,
            z: 34.0,
            body: Material::Metal,
        },
        Car {
            x: 19.4,
            z: 29.0,
            body: Material::Fabric,
        },
        Car {
            x: 19.4,
            z: 34.0,
            body: Material::Plaster,
        },
    ]
}

fn ground(world: &mut VoxelWorld) {
    // One walking surface. A hole in it is a hole: there is no soil sheet under the plot.
    fill_ground(world, 0.0, 32.6, 0.0, 39.2, Material::Grass);
}

fn sidewalks(world: &mut VoxelWorld) {
    fill_ground(world, 5.6, 27.0, 5.6, 22.8, Material::Stone);
    fill_ground(world, 7.0, 25.6, 7.0, 21.4, Material::Grass);
    fill_ground(world, 12.3, 13.5, 27.4, 38.0, Material::Stone);
    fill_ground(world, 19.1, 20.3, 27.4, 38.0, Material::Stone);
}

fn road(world: &mut VoxelWorld) {
    // Approach from the south, then a pad at the square. North of the pad is lawn.
    fill_ground(world, 13.5, 19.1, 22.8, 38.0, Material::Concrete);
    fill_ground(world, 10.5, 22.1, 19.6, 22.8, Material::Concrete);
}

fn fill_ground(world: &mut VoxelWorld, x0: f32, x1: f32, z0: f32, z1: f32, material: Material) {
    let block = world.block_size;
    let (xa, xb) = cell_span(x0, x1, block);
    let (za, zb) = cell_span(z0, z1, block);
    world.fill(xa, 0, za, xb, 0, zb, material, true);
}

fn house(world: &mut VoxelWorld, villa: Villa, block: f32) {
    let grid = grid(villa, block);
    let shell = shell_for(&grid, block);
    let roof = shell.wall_top + 1;
    let overhang = ncells(0.25, block).clamp(1, 3);
    for lx in 0..grid.w {
        for lz in 0..grid.d {
            let (x, z) = map(villa.face, &grid, lx, lz);
            world.set(x, 0, z, Material::Wood, true);
        }
    }
    for lx in -overhang..=grid.w - 1 + overhang {
        for lz in -overhang..=grid.d - 1 + overhang {
            let (x, z) = map(villa.face, &grid, lx, lz);
            world.set(x, roof, z, Material::Roof, false);
        }
    }
    for lx in 0..grid.w {
        for lz in 0..grid.d {
            let edge = lx == 0 || lz == 0 || lx == grid.w - 1 || lz == grid.d - 1;
            if !edge {
                continue;
            }
            for y in 1..=shell.wall_top {
                let (x, z) = map(villa.face, &grid, lx, lz);
                match skin(lx, y, lz, &shell) {
                    Skin::Open => world.carve(x, y, z, x, y, z),
                    Skin::Glass => world.set(x, y, z, Material::Glass, false),
                    Skin::Metal => world.set(x, y, z, Material::Metal, false),
                    Skin::Wall => world.set(x, y, z, Material::Plaster, false),
                }
            }
        }
    }
    for lx in 0..grid.w {
        for lz in 0..grid.d {
            if lz == grid.d - 1 {
                continue;
            }
            let rim = lx == 0 || lz == 0 || lx == grid.w - 1;
            if !rim {
                continue;
            }
            let (x, z) = map(villa.face, &grid, lx, lz);
            world.set(x, roof + 1, z, Material::Plaster, false);
        }
    }
}

enum Skin {
    Open,
    Glass,
    Metal,
    Wall,
}

fn skin(lx: i32, y: i32, lz: i32, shell: &Shell) -> Skin {
    let front = lz == shell.d - 1;
    let door =
        front && (shell.door_x0..=shell.door_x1).contains(&lx) && (1..=shell.door_h).contains(&y);
    if door {
        return Skin::Open;
    }
    if front
        && (1..=shell.door_h).contains(&y)
        && (lx == shell.door_x0 - 1 || lx == shell.door_x1 + 1)
    {
        return Skin::Metal;
    }
    if front && y == shell.door_h + 1 && (shell.door_x0 - 1..=shell.door_x1 + 1).contains(&lx) {
        return Skin::Metal;
    }
    let window_band = (shell.win_y0..=shell.win_y1).contains(&y);
    let beside_door = front
        && ((lx >= 1 && lx < shell.door_x0 - 1) || (lx > shell.door_x1 + 1 && lx <= shell.w - 2));
    let side = (lx == 0 || lx == shell.w - 1) && lz >= shell.d / 3 && lz <= (shell.d * 2) / 3;
    let back = lz == 0 && lx >= shell.w / 3 && lx <= (shell.w * 2) / 3;
    if window_band && (beside_door || side || back) {
        return Skin::Glass;
    }
    if front && (shell.door_h + 2..=shell.wall_top).contains(&y) && lx >= 1 && lx <= shell.w - 2 {
        if shell.mullion > 1 && lx % shell.mullion == 0 {
            return Skin::Metal;
        }
        return Skin::Glass;
    }
    Skin::Wall
}

fn shell_for(grid: &Grid, block: f32) -> Shell {
    let wall_top = wall_top(block);
    let door_w = ncells(DOOR_W_M, block).clamp(1, (grid.w - 2).max(1));
    let door_x0 = ((grid.w - door_w) / 2).max(1);
    let door_x1 = (door_x0 + door_w - 1).min(grid.w - 2);
    let door_h = ncells(DOOR_H_M, block).clamp(1, wall_top - 2);
    let win_y0 = ncells(1.05, block).clamp(1, door_h);
    let win_y1 = ncells(1.85, block).clamp(win_y0, door_h);
    Shell {
        w: grid.w,
        d: grid.d,
        door_h,
        door_x0,
        door_x1,
        wall_top,
        win_y0,
        win_y1,
        mullion: ncells(1.0, block).max(2),
    }
}

fn wall_top(block: f32) -> i32 {
    ncells(WALL_M, block).max(ncells(DOOR_H_M, block) + 2)
}

fn grid(villa: Villa, block: f32) -> Grid {
    let (ox, oz, w, d) = match villa.face {
        Face::South | Face::North => {
            let (ox, w) = cover(villa.x, HOUSE_W_M, block);
            let (oz, d) = cover(villa.z, HOUSE_D_M, block);
            (ox, oz, w, d)
        }
        Face::East | Face::West => {
            let (ox, d) = cover(villa.x, HOUSE_D_M, block);
            let (oz, w) = cover(villa.z, HOUSE_W_M, block);
            (ox, oz, w, d)
        }
    };
    Grid { ox, oz, w, d }
}

fn map(face: Face, grid: &Grid, lx: i32, lz: i32) -> (i32, i32) {
    match face {
        Face::South => (grid.ox + lx, grid.oz + lz),
        Face::North => (grid.ox + lx, grid.oz + (grid.d - 1 - lz)),
        Face::East => (grid.ox + lz, grid.oz + lx),
        Face::West => (grid.ox + (grid.d - 1 - lz), grid.oz + lx),
    }
}

fn tree(world: &mut VoxelWorld, x: f32, z: f32) {
    let block = world.block_size;
    let ix = (x / block).floor() as i32;
    let iz = (z / block).floor() as i32;
    if !world
        .blocks
        .get(&IVec3::new(ix, 0, iz))
        .is_some_and(|block| block.material == Material::Grass)
    {
        return;
    }
    let trunk = ncells(TREE_H_M, block);
    let thick = (0.28 / block).round().clamp(1.0, 3.0) as i32;
    let half = thick / 2;
    for dx in 0..thick {
        for dz in 0..thick {
            for y in 1..=trunk {
                world.set(ix - half + dx, y, iz - half + dz, Material::Trunk, y == 1);
            }
        }
    }
    // Two leafy disks. A filled ball at this resolution is tens of thousands of cells.
    disk(
        world,
        ix,
        iz,
        trunk - ncells(1.3, block),
        ncells(1.35, block),
    );
    disk(
        world,
        ix,
        iz,
        trunk - ncells(0.25, block),
        ncells(1.05, block),
    );
}

fn disk(world: &mut VoxelWorld, x: i32, z: i32, y: i32, radius: i32) {
    if y < 2 || radius < 1 {
        return;
    }
    let r2 = radius * radius;
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            if dx * dx + dz * dz > r2 {
                continue;
            }
            let cell = IVec3::new(x + dx, y, z + dz);
            if world.blocks.contains_key(&cell) {
                continue;
            }
            world.set(cell.x, cell.y, cell.z, Material::Leaf, false);
        }
    }
}

fn park(world: &mut VoxelWorld, car: Car) {
    let block = world.block_size;
    let (x0, nx) = cover(car.x, CAR_W_M, block);
    let (z0, nz) = cover(car.z, CAR_L_M, block);
    let x1 = x0 + nx - 1;
    let z1 = z0 + nz - 1;
    let roof = ncells(CAR_H_M, block).max(3);
    for x in x0..=x1 {
        for z in z0..=z1 {
            let edge = x == x0 || x == x1 || z == z0 || z == z1;
            world.set(x, 1, z, Material::WoodDark, false);
            if edge {
                for y in 2..roof {
                    world.set(x, y, z, car.body, false);
                }
            }
            world.set(x, roof, z, car.body, false);
        }
    }
    let glass_y = (roof * 2 / 3).clamp(2, roof - 1);
    let wx0 = x0 + nx / 5;
    let wx1 = x1 - nx / 5;
    for x in wx0..=wx1 {
        world.set(x, glass_y, z1, Material::Glass, false);
    }
}

fn ncells(metres: f32, block: f32) -> i32 {
    (metres / block).round().max(1.0) as i32
}

fn cover(start: f32, size: f32, block: f32) -> (i32, i32) {
    let i0 = (start / block).floor() as i32;
    let i1 = ((start + size) / block - 1.0e-3).floor() as i32;
    (i0, (i1 - i0 + 1).max(1))
}

fn cell_span(a: f32, b: f32, block: f32) -> (i32, i32) {
    let i0 = (a / block).floor() as i32;
    let i1 = ((b / block) - 1.0e-3).floor() as i32;
    (i0.min(i1), i0.max(i1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_villas_face_a_square_and_the_road_stops() {
        for block in [0.1_f32, 0.2] {
            let mut world = VoxelWorld::new(block);
            let spawn = build(&mut world);
            assert_eq!(villas().len(), 10);
            assert!((spawn.feet.x - SPAWN_X).abs() < 1.0e-4);
            assert!((spawn.feet.y - block).abs() < 1.0e-4);

            for villa in villas() {
                let grid = grid(villa, block);
                let shell = shell_for(&grid, block);
                let lx = shell.door_x0 + (shell.door_x1 - shell.door_x0) / 2;
                let (dx, dz) = map(villa.face, &grid, lx, grid.d - 1);
                let mut open = 0;
                let mut y = 1;
                while y < 80 && !world.blocks.contains_key(&IVec3::new(dx, y, dz)) {
                    open += 1;
                    y += 1;
                }
                let clearance = open as f32 * block;
                assert!(
                    (2.0..=2.45).contains(&clearance),
                    "door clearance {clearance:.2} m at block {block}"
                );
                let lintel = world.blocks.get(&IVec3::new(dx, y, dz));
                assert!(lintel.is_some(), "door has no lintel");
                let width = grid.w as f32 * block;
                assert!(
                    (5.4..=6.6).contains(&width),
                    "house is {width:.2} m wide at block {block}"
                );
                let (rx, rz) = map(villa.face, &grid, 2, 2);
                let roof = world
                    .blocks
                    .get(&IVec3::new(rx, shell.wall_top + 1, rz))
                    .expect("villa roof");
                assert_eq!(roof.material, Material::Roof);
                assert!(!roof.anchored);
            }

            let mut north_road = i32::MAX;
            let mut south_road = i32::MIN;
            let south_band = (SPAWN_Z / block).floor() as i32;
            let mut west_lane = i32::MAX;
            let mut east_lane = i32::MIN;
            for (cell, block_cell) in &world.blocks {
                if block_cell.material == Material::Concrete && cell.y == 0 {
                    north_road = north_road.min(cell.z);
                    south_road = south_road.max(cell.z);
                    if cell.z == south_band {
                        west_lane = west_lane.min(cell.x);
                        east_lane = east_lane.max(cell.x);
                    }
                }
            }
            let length = (south_road - north_road + 1) as f32 * block;
            let width = (east_lane - west_lane + 1) as f32 * block;
            assert!(length > 12.0, "road is {length:.1} m at block {block}");
            assert!(
                (5.2..=6.6).contains(&width),
                "road is {width:.2} m wide at block {block}"
            );
            let lawn_x = (SPAWN_X / block).floor() as i32;
            let lawn_z = (12.0 / block).floor() as i32;
            let lawn = world
                .blocks
                .get(&IVec3::new(lawn_x, 0, lawn_z))
                .expect("lawn in the square");
            assert_eq!(lawn.material, Material::Grass);
            assert!(lawn_z < north_road, "the road does not stop at the square");

            let mut trunks = 0;
            for (x, z) in trees() {
                let ix = (x / block).floor() as i32;
                let iz = (z / block).floor() as i32;
                let mut top = 0;
                for y in 1..200 {
                    if world
                        .blocks
                        .get(&IVec3::new(ix, y, iz))
                        .is_some_and(|cell| cell.material == Material::Trunk)
                    {
                        top = y;
                    }
                }
                if top > 0 {
                    trunks += 1;
                    let crown = top as f32 * block;
                    assert!(crown >= 4.8, "tree is only {crown:.2} m at block {block}");
                }
            }
            assert!(trunks >= 8, "street trees planted: {trunks}");

            for car in cars() {
                let (x0, nx) = cover(car.x, CAR_W_M, block);
                let (z0, nz) = cover(car.z, CAR_L_M, block);
                let roof = ncells(CAR_H_M, block).max(3);
                let body = world
                    .blocks
                    .get(&IVec3::new(x0 + nx / 2, roof, z0 + nz / 2))
                    .expect("car roof");
                assert_eq!(body.material, car.body);
                let length = nz as f32 * block;
                let height = roof as f32 * block;
                assert!((4.0..=4.8).contains(&length), "car is {length:.2} m long");
                assert!((1.3..=1.7).contains(&height), "car roof is {height:.2} m");
            }

            assert!(
                world.blocks.len() < 350_000,
                "street has {} blocks at {block} m",
                world.blocks.len()
            );
        }
    }
}
