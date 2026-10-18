//! A two-storey villa. Blocks are 0.25 m. The east wing slab is carried by walls and columns down
//! to the anchored ground course. Sever that path and the wing falls. The south face is a glass
//! pavilion with a solid bedroom wing beside it — the usual modern-villa section, not a decorated box.

use crate::material::Material;
use crate::math3::Vec3;
use crate::world::VoxelWorld;

pub struct Spawn {
    pub feet: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

pub fn build(world: &mut VoxelWorld) -> Spawn {
    let s = world.block_size;
    // House local origin, in cells. Lawn surrounds it.
    let ox = 24;
    let oz = 28;
    // 14.0 m × 10.0 m footprint.
    let w = 56;
    let d = 40;

    lawn(world, ox, oz, w, d);
    terrace(world, ox, oz, w, d);
    shell(world, ox, oz, w, d);
    glass_pavilion(world, ox, oz, w, d);
    east_wing(world, ox, oz);
    carport(world, ox, oz);
    interior(world, ox, oz);
    furniture(world, ox, oz);
    perimeter_fence(world, ox, oz, w, d);

    // South terrace, facing the glass wall (yaw 0 looks toward -Z).
    let feet = Vec3::new((ox + 22) as f32 * s, s, (oz + d + 8) as f32 * s);
    Spawn {
        feet,
        yaw: 0.0,
        pitch: -0.08,
    }
}

fn plot_bounds(ox: i32, oz: i32, w: i32, d: i32) -> (i32, i32, i32, i32) {
    (-4, ox + w + 28, -4, oz + d + 36)
}

fn lawn(world: &mut VoxelWorld, ox: i32, oz: i32, w: i32, d: i32) {
    let (x0, x1, z0, z1) = plot_bounds(ox, oz, w, d);
    // Walkable ground. Anchored so the house stands on it, but not indestructible:
    // a blast removes it, and there is no hidden floor underneath.
    world.fill(x0, 0, z0, x1, 0, z1, Material::Grass, true);
    world.fill(x0, -1, z0, x1, -1, z1, Material::Soil, true);
}

fn perimeter_fence(world: &mut VoxelWorld, ox: i32, oz: i32, w: i32, d: i32) {
    let (x0, x1, z0, z1) = plot_bounds(ox, oz, w, d);
    // 0.5 m thick, 3 m above the walking surface. Jump apex is about 1.1 m, so this is a wall,
    // not a curb. Footing includes the ground course so a blast cannot open a gap under it.
    let top = 12;
    world.fill_barrier(x0, -1, z0, x0 + 1, top, z1, Material::Metal);
    world.fill_barrier(x1 - 1, -1, z0, x1, top, z1, Material::Metal);
    world.fill_barrier(x0, -1, z0, x1, top, z0 + 1, Material::Metal);
    world.fill_barrier(x0, -1, z1 - 1, x1, top, z1, Material::Metal);
}

fn terrace(world: &mut VoxelWorld, ox: i32, oz: i32, w: i32, d: i32) {
    world.fill(
        ox - 6,
        0,
        oz + d - 1,
        ox + w + 6,
        0,
        oz + d + 18,
        Material::Stone,
        true,
    );
}

fn shell(world: &mut VoxelWorld, ox: i32, oz: i32, w: i32, d: i32) {
    // Interior floor finish, still the anchored ground course.
    world.fill(
        ox + 1,
        0,
        oz + 1,
        ox + w - 2,
        0,
        oz + d - 2,
        Material::Wood,
        true,
    );

    // Concrete columns, 0.5 m, full height of the glass pavilion plus the roof.
    let cols = [
        (0, 0),
        (0, d - 2),
        (34, 0),
        (34, d - 2),
        (w - 2, 0),
        (w - 2, d - 2),
        (16, d - 2),
        (16, 0),
    ];
    for (x, z) in cols {
        world.fill(
            ox + x,
            1,
            oz + z,
            ox + x + 1,
            24,
            oz + z + 1,
            Material::Concrete,
            false,
        );
    }

    // Perimeter plaster, two storeys. Openings are carved after.
    world.fill(ox, 1, oz, ox + w - 1, 23, oz + 1, Material::Plaster, false);
    world.fill(
        ox,
        1,
        oz + d - 2,
        ox + w - 1,
        23,
        oz + d - 1,
        Material::Plaster,
        false,
    );
    world.fill(ox, 1, oz, ox + 1, 23, oz + d - 1, Material::Plaster, false);
    world.fill(
        ox + w - 2,
        1,
        oz,
        ox + w - 1,
        23,
        oz + d - 1,
        Material::Plaster,
        false,
    );
    // Dense concrete skin outside the plaster, so the wall is thicker than one face.
    world.fill(
        ox - 1,
        1,
        oz - 1,
        ox - 1,
        23,
        oz + d,
        Material::Concrete,
        false,
    );
    world.fill(
        ox + w,
        1,
        oz - 1,
        ox + w,
        23,
        oz + d,
        Material::Concrete,
        false,
    );
    world.fill(
        ox - 1,
        1,
        oz - 1,
        ox + w,
        23,
        oz - 1,
        Material::Concrete,
        false,
    );

    // North entrance, 1.5 m × 2.25 m.
    world.carve(ox + 22, 1, oz - 1, ox + 27, 9, oz + 1);
    // North windows.
    window(world, ox + 6, oz, 8, 4, true);
    window(world, ox + 40, oz, 8, 4, true);
    window(world, ox + 40, oz, 8, 16, true);
    // West punched windows.
    window(world, ox, oz + 8, 6, 5, false);
    window(world, ox, oz + 24, 6, 5, false);
    // East windows, both floors.
    window_east(world, ox + w - 2, oz + 8, 6, 5);
    window_east(world, ox + w - 2, oz + 22, 6, 5);
    window_east(world, ox + w - 2, oz + 8, 6, 16);
    window_east(world, ox + w - 2, oz + 22, 6, 16);

    // Roof slab and dark fascia. The overhang on the south is connected only through this slab.
    world.fill(
        ox - 2,
        24,
        oz - 1,
        ox + w + 1,
        24,
        oz + d + 3,
        Material::Roof,
        false,
    );
    world.fill(
        ox - 3,
        24,
        oz + d + 2,
        ox + w + 2,
        24,
        oz + d + 4,
        Material::Metal,
        false,
    );
    // Parapet, omitted on the south overhang so the glass line stays clean.
    world.fill(
        ox - 1,
        25,
        oz - 1,
        ox + w,
        27,
        oz - 1,
        Material::Plaster,
        false,
    );
    world.fill(
        ox - 1,
        25,
        oz,
        ox - 1,
        27,
        oz + d - 1,
        Material::Plaster,
        false,
    );
    world.fill(
        ox + w,
        25,
        oz,
        ox + w,
        27,
        oz + d - 1,
        Material::Plaster,
        false,
    );
    // Stair bulkhead on the roof.
    world.fill(
        ox + 42,
        25,
        oz + 4,
        ox + 50,
        29,
        oz + 14,
        Material::Plaster,
        false,
    );
    world.fill(
        ox + 44,
        30,
        oz + 6,
        ox + 48,
        30,
        oz + 10,
        Material::Glass,
        false,
    );
}

fn glass_pavilion(world: &mut VoxelWorld, ox: i32, oz: i32, _w: i32, d: i32) {
    // Double-height living room: no upper slab west of the party wall.
    // South wall of the pavilion becomes glass from floor to roof, with metal mullions.
    world.carve(ox + 2, 1, oz + d - 2, ox + 33, 23, oz + d - 1);
    for x in (ox + 2..=ox + 32).step_by(4) {
        world.fill(x, 1, oz + d - 2, x, 23, oz + d - 1, Material::Metal, false);
    }
    world.fill(
        ox + 2,
        1,
        oz + d - 2,
        ox + 32,
        1,
        oz + d - 1,
        Material::Metal,
        false,
    );
    world.fill(
        ox + 2,
        23,
        oz + d - 2,
        ox + 32,
        23,
        oz + d - 1,
        Material::Metal,
        false,
    );
    world.fill(
        ox + 3,
        2,
        oz + d - 2,
        ox + 32,
        22,
        oz + d - 1,
        Material::Glass,
        false,
    );
    // Mullions punch back through the glass.
    for x in (ox + 2..=ox + 32).step_by(4) {
        world.fill(x, 1, oz + d - 2, x, 23, oz + d - 1, Material::Metal, false);
    }
    // Sliding door, so the terrace spawn can walk in.
    world.carve(ox + 18, 1, oz + d - 2, ox + 25, 9, oz + d - 1);
    world.fill(
        ox + 18,
        1,
        oz + d - 2,
        ox + 25,
        1,
        oz + d - 1,
        Material::Metal,
        false,
    );
    world.fill(
        ox + 18,
        9,
        oz + d - 2,
        ox + 25,
        9,
        oz + d - 1,
        Material::Metal,
        false,
    );
}

fn east_wing(world: &mut VoxelWorld, ox: i32, oz: i32) {
    // Party wall between the double-height room and the bedroom wing. This wall, plus the east
    // perimeter, is what holds the upper slab. It is not anchored.
    world.fill(
        ox + 34,
        1,
        oz + 2,
        ox + 35,
        23,
        oz + 37,
        Material::Concrete,
        false,
    );
    world.carve(ox + 34, 1, oz + 16, ox + 35, 9, oz + 20); // ground door
    world.carve(ox + 34, 13, oz + 18, ox + 35, 21, oz + 21); // upper door

    // Upper slab, east wing only, with a stair void.
    world.fill(
        ox + 36,
        12,
        oz + 1,
        ox + 54,
        12,
        oz + 37,
        Material::Concrete,
        false,
    );
    world.carve(ox + 42, 12, oz + 4, ox + 48, 12, oz + 14);
    // Timber finish on that slab.
    world.fill(
        ox + 36,
        13,
        oz + 1,
        ox + 54,
        13,
        oz + 37,
        Material::Wood,
        false,
    );
    world.carve(ox + 42, 13, oz + 4, ox + 48, 13, oz + 14);

    // Bedroom partition and bathroom.
    world.fill(
        ox + 36,
        14,
        oz + 20,
        ox + 53,
        23,
        oz + 21,
        Material::Plaster,
        false,
    );
    world.carve(ox + 40, 14, oz + 20, ox + 43, 21, oz + 21);
    world.fill(
        ox + 50,
        14,
        oz + 2,
        ox + 51,
        23,
        oz + 16,
        Material::Plaster,
        false,
    );

    // South balcony, a cantilever off the east slab. Supported only by that connection.
    world.fill(
        ox + 38,
        12,
        oz + 38,
        ox + 52,
        12,
        oz + 46,
        Material::Concrete,
        false,
    );
    world.fill(
        ox + 38,
        13,
        oz + 38,
        ox + 52,
        13,
        oz + 46,
        Material::Wood,
        false,
    );
    world.fill(
        ox + 38,
        14,
        oz + 46,
        ox + 52,
        16,
        oz + 46,
        Material::Glass,
        false,
    );
    world.fill(
        ox + 38,
        14,
        oz + 38,
        ox + 38,
        16,
        oz + 46,
        Material::Glass,
        false,
    );
    world.fill(
        ox + 52,
        14,
        oz + 38,
        ox + 52,
        16,
        oz + 46,
        Material::Glass,
        false,
    );
}

fn carport(world: &mut VoxelWorld, ox: i32, oz: i32) {
    // West carport. Four columns, a thin roof. Shoot the columns and the roof falls.
    for (x, z) in [
        (ox - 14, oz + 8),
        (ox - 14, oz + 22),
        (ox - 2, oz + 8),
        (ox - 2, oz + 22),
    ] {
        world.fill(x, 1, z, x + 1, 11, z + 1, Material::Concrete, false);
    }
    world.fill(
        ox - 16,
        12,
        oz + 6,
        ox - 1,
        12,
        oz + 26,
        Material::Concrete,
        false,
    );
    world.fill(
        ox - 16,
        13,
        oz + 6,
        ox - 1,
        13,
        oz + 26,
        Material::Roof,
        false,
    );
}

fn interior(world: &mut VoxelWorld, ox: i32, oz: i32) {
    // Straight stair against the party wall, rising toward +Z then turning is more than this
    // grid wants. A single flight along +Y / +Z is readable and walkable as blocks.
    for step in 0..12 {
        let y = 1 + step;
        let z = oz + 4 + step;
        world.fill(ox + 42, y, z, ox + 47, y, z, Material::WoodDark, false);
    }
    // Bathroom tile.
    world.fill(
        ox + 51,
        14,
        oz + 3,
        ox + 53,
        14,
        oz + 12,
        Material::Tile,
        false,
    );
}

fn furniture(world: &mut VoxelWorld, ox: i32, oz: i32) {
    // Kitchen island, ground floor, west.
    world.fill(
        ox + 6,
        1,
        oz + 6,
        ox + 14,
        3,
        oz + 10,
        Material::Wood,
        false,
    );
    world.fill(
        ox + 6,
        4,
        oz + 6,
        ox + 14,
        4,
        oz + 10,
        Material::Stone,
        false,
    );
    // Sofa facing the glass.
    world.fill(
        ox + 8,
        1,
        oz + 26,
        ox + 18,
        2,
        oz + 29,
        Material::Fabric,
        false,
    );
    world.fill(
        ox + 8,
        3,
        oz + 28,
        ox + 18,
        3,
        oz + 29,
        Material::Fabric,
        false,
    );
    // Dining table.
    world.fill(
        ox + 20,
        1,
        oz + 18,
        ox + 28,
        3,
        oz + 22,
        Material::WoodDark,
        false,
    );
    // Bed, upper floor.
    world.fill(
        ox + 38,
        14,
        oz + 26,
        ox + 48,
        15,
        oz + 34,
        Material::Linen,
        false,
    );
    world.fill(
        ox + 38,
        14,
        oz + 33,
        ox + 48,
        17,
        oz + 34,
        Material::WoodDark,
        false,
    );
    // Hedge, north boundary of the plot. A modern villa still has a edge.
    world.fill(20, 1, 2, 96, 3, 3, Material::Grass, false);
}

fn window(world: &mut VoxelWorld, x: i32, z: i32, width: i32, y0: i32, north: bool) {
    let z1 = if north { z + 1 } else { z };
    let _ = z1;
    let z0 = if north { z - 1 } else { z };
    world.carve(x, y0, z0, x + width - 1, y0 + 5, z + 1);
    world.fill(x, y0, z, x + width - 1, y0, z + 1, Material::Metal, false);
    world.fill(
        x,
        y0 + 5,
        z,
        x + width - 1,
        y0 + 5,
        z + 1,
        Material::Metal,
        false,
    );
    world.fill(x, y0, z, x, y0 + 5, z + 1, Material::Metal, false);
    world.fill(
        x + width - 1,
        y0,
        z,
        x + width - 1,
        y0 + 5,
        z + 1,
        Material::Metal,
        false,
    );
    world.fill(
        x + 1,
        y0 + 1,
        z,
        x + width - 2,
        y0 + 4,
        z + 1,
        Material::Glass,
        false,
    );
}

fn window_east(world: &mut VoxelWorld, x: i32, z: i32, depth: i32, y0: i32) {
    world.carve(x, y0, z, x + 1, y0 + 5, z + depth - 1);
    world.fill(x, y0, z, x + 1, y0, z + depth - 1, Material::Metal, false);
    world.fill(
        x,
        y0 + 5,
        z,
        x + 1,
        y0 + 5,
        z + depth - 1,
        Material::Metal,
        false,
    );
    world.fill(x, y0, z, x + 1, y0 + 5, z, Material::Metal, false);
    world.fill(
        x,
        y0,
        z + depth - 1,
        x + 1,
        y0 + 5,
        z + depth - 1,
        Material::Metal,
        false,
    );
    world.fill(
        x,
        y0 + 1,
        z + 1,
        x + 1,
        y0 + 4,
        z + depth - 2,
        Material::Glass,
        false,
    );
}
