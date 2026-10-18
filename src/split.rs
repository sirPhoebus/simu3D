//! Cubic division of a block.
//!
//! Ten is not a product of three equal integers, so a cube cannot be cut into ten equal cubes.
//! The blast uses the nearest cubic grid: n = round(cbrt(count)), clamped to 1..=3. Count 10
//! becomes 8 cubes of side/2. Each piece has equal extents. A thin slab reads as a face, not a voxel.

use crate::math3::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Piece {
    /// Centre, relative to the parent block's centre.
    pub local_center: Vec3,
    pub half: Vec3,
}

impl Piece {
    pub fn volume(self) -> f32 {
        8.0 * self.half.x * self.half.y * self.half.z
    }

    pub fn is_cube(self) -> bool {
        (self.half.x - self.half.y).abs() < 1.0e-5 && (self.half.x - self.half.z).abs() < 1.0e-5
    }
}

/// 10 cm³. This edge is not a setting and does not follow the parent box.
pub fn small_cube_edge() -> f32 {
    1.0e-5_f32.cbrt()
}

/// Pack the parent with the fixed small cubes, centred. At most 10 per axis.
/// A different box size changes how many fit. It does not change their edge.
pub fn fixed_small_cubes(parent: Vec3) -> Vec<Piece> {
    let edge = small_cube_edge();
    let n = ((parent.x.min(parent.y).min(parent.z) / edge).round() as usize).clamp(1, 10);
    let half = Vec3::new(edge, edge, edge) * 0.5;
    let span = edge * n as f32;
    let origin = Vec3::new(-span, -span, -span) * 0.5;
    let mut out = Vec::with_capacity(n * n * n);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let min = origin + Vec3::new(x as f32 * edge, y as f32 * edge, z as f32 * edge);
                out.push(Piece {
                    local_center: min + half,
                    half,
                });
            }
        }
    }
    out
}

/// Grid resolution. 1000 pieces is 10×10×10. A 10×10 sheet would be faces, so the third axis is divided too.
pub fn grid_n(count: usize) -> usize {
    let n = (count.max(1) as f32).cbrt().round() as usize;
    n.clamp(1, 10)
}

/// Equal cubes. A cubic parent is tiled with no gap and no overlap. A non-cube parent
/// gets a cube of side min(cell) centred in each grid cell, so the piece is still a cube.
pub fn cubes(size: Vec3, count: usize) -> Vec<Piece> {
    let n = grid_n(count);
    let nf = n as f32;
    let cell = Vec3::new(size.x / nf, size.y / nf, size.z / nf);
    let side = cell.x.min(cell.y).min(cell.z);
    let half = Vec3::new(side, side, side) * 0.5;
    let origin = size * -0.5;
    let mut out = Vec::with_capacity(n * n * n);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let min = Vec3::new(
                    origin.x + x as f32 * cell.x,
                    origin.y + y as f32 * cell.y,
                    origin.z + z as f32 * cell.z,
                );
                out.push(Piece {
                    local_center: min + cell * 0.5,
                    half,
                });
            }
        }
    }
    out
}

pub fn pieces(size: Vec3, count: usize, min_extent: f32) -> Vec<Piece> {
    let pieces = cubes(size, count);
    if pieces[0].half.x * 2.0 < min_extent {
        vec![Piece {
            local_center: Vec3::ZERO,
            half: size * 0.5,
        }]
    } else {
        pieces
    }
}

/// Which of the 10×10×10 small cubes the blast frees. The rest stay in the parent.
/// `blast_from_center` points from the parent centre toward the charge.
/// A sub-cube leaves when `parent_impulse * exposure` exceeds `resistance`.
/// Exposure is 1 on the face toward the charge and 0 on the far face, so the
/// crater shape follows the force and the material. At most 350 pieces leave.
pub fn partial_break(
    size: Vec3,
    blast_from_center: Vec3,
    parent_impulse: f32,
    resistance: f32,
    count: usize,
    min_extent: f32,
) -> (Vec<Piece>, Vec<Piece>) {
    let pieces = pieces(size, count, min_extent);
    let len = blast_from_center.length();
    let dir = if len < 1.0e-4 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        blast_from_center * (1.0 / len)
    };
    let reach = size.x.max(size.y).max(size.z) * 0.5;
    let mut ejected = Vec::new();
    let mut kept = Vec::new();
    for piece in pieces {
        let along = piece.local_center.dot(dir);
        let depth = ((reach - along) / (2.0 * reach)).clamp(0.0, 1.0);
        let exposure = (1.0 - depth).powi(2);
        if parent_impulse * exposure > resistance {
            ejected.push((exposure, piece));
        } else {
            kept.push(piece);
        }
    }
    ejected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if ejected.len() > 350 {
        for (_, piece) in ejected.drain(350..) {
            kept.push(piece);
        }
    }
    let ejected = ejected.into_iter().map(|(_, piece)| piece).collect();
    (ejected, kept)
}

pub fn overlap_volume(a: Piece, b: Piece) -> f32 {
    let a_min = a.local_center - a.half;
    let a_max = a.local_center + a.half;
    let b_min = b.local_center - b.half;
    let b_max = b.local_center + b.half;
    let overlap = (a_max.min(b_max) - a_min.max(b_min)).max(Vec3::ZERO);
    overlap.x * overlap.y * overlap.z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousand_cubes_tile_a_ten_centimetre_cube() {
        let side = 10.0 * 1.0e-5_f32.cbrt();
        let size = Vec3::new(side, side, side);
        let pieces = fixed_small_cubes(size);
        assert_eq!(pieces.len(), 1000);
        assert!(pieces.iter().all(|p| p.is_cube()));
        let small = pieces[0].volume();
        assert!((pieces[0].half.x * 2.0 - small_cube_edge()).abs() < 1.0e-6);
        assert!(
            (small - 1.0e-5).abs() < 1.0e-8,
            "small cube {small} m³, expected 10 cm³"
        );
        let sum: f32 = pieces.iter().map(|p| p.volume()).sum();
        let parent = size.x * size.y * size.z;
        assert!((sum - parent).abs() < 1.0e-6, "volume {sum} vs {parent}");
        for i in 0..pieces.len() {
            for j in (i + 1)..pieces.len() {
                let overlap = overlap_volume(pieces[i], pieces[j]);
                assert!(overlap < 1.0e-9, "overlap {overlap} between {i} and {j}");
            }
        }
    }

    #[test]
    fn a_different_box_does_not_resize_the_small_cube() {
        let pieces = fixed_small_cubes(Vec3::new(0.40, 0.40, 0.40));
        let edge = small_cube_edge();
        assert!(!pieces.is_empty());
        assert!(pieces.len() <= 1000);
        assert!(pieces
            .iter()
            .all(|p| (p.half.x * 2.0 - edge).abs() < 1.0e-6));
    }

    #[test]
    fn fragment_count_and_minimum_extent_control_subdivision() {
        let size = Vec3::new(0.2, 0.2, 0.2);
        assert_eq!(pieces(size, 64, 0.0).len(), 64);
        let unsplit = pieces(size, 1000, 0.03);
        assert_eq!(unsplit.len(), 1);
        assert!((unsplit[0].volume() - size.x * size.y * size.z).abs() < 1.0e-6);
    }
}
