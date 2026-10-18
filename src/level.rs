//! Which map the window loads. The settings panel cycles the id and rebuilds.

use crate::biome::{self, Terrain};
use crate::street;
use crate::villa::{self, Spawn};
use crate::world::VoxelWorld;

pub const COUNT: u8 = 3;

pub fn normalize(id: u8) -> u8 {
    id % COUNT
}

pub fn name(id: u8) -> &'static str {
    match normalize(id) {
        0 => "VILLA",
        1 => "BIOME",
        _ => "STREET",
    }
}

pub fn blurb(id: u8) -> &'static str {
    match normalize(id) {
        0 => "The metal fence does not break. The ground does: blow a hole and you fall through it. There is no floor under the plot.",
        1 => "Biome: 1 km² of hills. The ground does not break. Trees, bushes, and rocks do.",
        _ => "Dead-end street. Ten villas face a square. Trees and cars line the road. The pavement breaks; blow a hole and you fall through.",
    }
}

pub fn build(id: u8, world: &mut VoxelWorld) -> (Spawn, Option<Terrain>) {
    match normalize(id) {
        0 => (villa::build(world), None),
        1 => {
            let (spawn, terrain) = biome::build(world);
            (spawn, Some(terrain))
        }
        _ => (street::build(world), None),
    }
}
