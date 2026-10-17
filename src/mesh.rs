//! One closed cube per block. An exposed-face mesh puts every wall cube on the
//! same plane, so the wall is a single face. Insetting each face on its own
//! disconnects the six faces into plates. The cube is shrunk as a whole, and
//! all six faces are emitted so the edges meet.

use crate::material::Material;
use crate::math3::Vec3;
use crate::world::VoxelWorld;

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

const AXES: [(usize, bool); 6] = [
    (0, true),
    (0, false),
    (1, true),
    (1, false),
    (2, true),
    (2, false),
];

pub fn build(world: &VoxelWorld, transparent_pass: bool) -> Mesh {
    let mut mesh = Mesh::default();
    let n = world.blocks.len();
    mesh.vertices.reserve(n * 24);
    mesh.indices.reserve(n * 36);
    let s = world.block_size;
    // Uniform shrink. The gap is the joint. The six faces still meet.
    let inset = s * 0.06;
    let side = s - inset * 2.0;
    for (cell, block) in &world.blocks {
        if block.material.transparent() != transparent_pass {
            continue;
        }
        let min = world.cell_min(*cell);
        let origin = Vec3::new(min.x + inset, min.y + inset, min.z + inset);
        emit_closed_cube(&mut mesh, origin, side, block.material);
    }
    mesh
}

fn emit_closed_cube(mesh: &mut Mesh, origin: Vec3, side: f32, material: Material) {
    let min = [origin.x, origin.y, origin.z];
    for (axis, positive) in AXES {
        emit_cube_face(mesh, min, axis, positive, side, material);
    }
}

fn emit_cube_face(mesh: &mut Mesh, min: [f32; 3], axis: usize, positive: bool, s: f32, material: Material) {
    let mut normal = [0.0, 0.0, 0.0];
    normal[axis] = if positive { 1.0 } else { -1.0 };
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let mut origin = min;
    if positive {
        origin[axis] += s;
    }
    let steps = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)];
    let mut corners = [[0.0; 3]; 4];
    for (i, (ou, ov)) in steps.iter().enumerate() {
        let mut p = origin;
        p[u] += ou;
        p[v] += ov;
        corners[i] = p;
    }
    let e1 = Vec3::new(
        corners[1][0] - corners[0][0],
        corners[1][1] - corners[0][1],
        corners[1][2] - corners[0][2],
    );
    let e2 = Vec3::new(
        corners[2][0] - corners[0][0],
        corners[2][1] - corners[0][1],
        corners[2][2] - corners[0][2],
    );
    let stored = Vec3::new(normal[0], normal[1], normal[2]);
    if e1.cross(e2).dot(stored) < 0.0 {
        corners.swap(1, 3);
    }
    let base = mesh.vertices.len() as u32;
    let color = material.color();
    for p in corners {
        mesh.vertices.push(Vertex {
            position: p,
            normal,
            color,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::world::VoxelWorld;

    #[test]
    fn one_block_is_a_closed_cube_not_a_face() {
        let mut world = VoxelWorld::new(1.0);
        world.set(0, 0, 0, Material::Concrete, false);
        let mesh = build(&world, false);
        assert_eq!(mesh.indices.len(), 36, "a cube has 6 faces, not one");
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(v.position[axis]);
                max[axis] = max[axis].max(v.position[axis]);
            }
        }
        let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        assert!(extent[0] > 0.5 && extent[1] > 0.5 && extent[2] > 0.5);
        let mid = (extent[0] + extent[1] + extent[2]) / 3.0;
        for axis in 0..3 {
            assert!(
                (extent[axis] - mid).abs() < 1.0e-4,
                "extent is not cubic: {extent:?}"
            );
        }
        let mut edges = std::collections::HashMap::<(i64, i64, i64, i64, i64, i64), i32>::new();
        for tri in mesh.indices.chunks(3) {
            for pair in [(0, 1), (1, 2), (2, 0)] {
                let a = quant(mesh.vertices[tri[pair.0] as usize].position);
                let b = quant(mesh.vertices[tri[pair.1] as usize].position);
                let key = if a <= b {
                    (a.0, a.1, a.2, b.0, b.1, b.2)
                } else {
                    (b.0, b.1, b.2, a.0, a.1, a.2)
                };
                *edges.entry(key).or_default() += 1;
            }
        }
        let shared = edges.values().filter(|n| **n >= 2).count();
        assert!(
            shared >= 12,
            "cube edges are not shared, so the faces do not meet: {shared}"
        );
    }

    fn quant(p: [f32; 3]) -> (i64, i64, i64) {
        (
            (p[0] * 1000.0).round() as i64,
            (p[1] * 1000.0).round() as i64,
            (p[2] * 1000.0).round() as i64,
        )
    }
}
