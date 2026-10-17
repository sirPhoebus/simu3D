//! Sparse block grid. One cell is one house block. Destruction removes cells; it does not keep a per-cell rigid body for the standing structure.

use std::collections::HashMap;

use crate::material::Material;
use crate::math3::{IVec3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub material: Material,
    pub anchored: bool,
    /// Blast and collapse leave this cell alone. Used for the perimeter fence.
    pub indestructible: bool,
}

#[derive(Clone, Debug)]
pub struct VoxelWorld {
    pub block_size: f32,
    pub blocks: HashMap<IVec3, Block>,
}

impl VoxelWorld {
    pub fn new(block_size: f32) -> Self {
        Self {
            block_size,
            blocks: HashMap::new(),
        }
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, material: Material, anchored: bool) {
        self.blocks.insert(
            IVec3::new(x, y, z),
            Block {
                material,
                anchored,
                indestructible: false,
            },
        );
    }

    pub fn fill(
        &mut self,
        x0: i32,
        y0: i32,
        z0: i32,
        x1: i32,
        y1: i32,
        z1: i32,
        material: Material,
        anchored: bool,
    ) {
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (y0, y1) = (y0.min(y1), y0.max(y1));
        let (z0, z1) = (z0.min(z1), z0.max(z1));
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    self.set(x, y, z, material, anchored);
                }
            }
        }
    }

    pub fn fill_barrier(
        &mut self,
        x0: i32,
        y0: i32,
        z0: i32,
        x1: i32,
        y1: i32,
        z1: i32,
        material: Material,
    ) {
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (y0, y1) = (y0.min(y1), y0.max(y1));
        let (z0, z1) = (z0.min(z1), z0.max(z1));
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    self.blocks.insert(
                        IVec3::new(x, y, z),
                        Block {
                            material,
                            anchored: true,
                            indestructible: true,
                        },
                    );
                }
            }
        }
    }

    pub fn carve(&mut self, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32) {
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (y0, y1) = (y0.min(y1), y0.max(y1));
        let (z0, z1) = (z0.min(z1), z0.max(z1));
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    self.blocks.remove(&IVec3::new(x, y, z));
                }
            }
        }
    }

    pub fn cell_min(&self, cell: IVec3) -> Vec3 {
        cell.to_vec3() * self.block_size
    }

    pub fn cell_center(&self, cell: IVec3) -> Vec3 {
        self.cell_min(cell) + Vec3::new(self.block_size, self.block_size, self.block_size) * 0.5
    }

    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        let mut iter = self.blocks.keys().copied();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for c in iter {
            min.x = min.x.min(c.x);
            min.y = min.y.min(c.y);
            min.z = min.z.min(c.z);
            max.x = max.x.max(c.x);
            max.y = max.y.max(c.y);
            max.z = max.z.max(c.z);
        }
        Some((min, max))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GreedyBox {
    pub min: IVec3,
    pub size: IVec3,
    pub material: Material,
}

/// Non-overlapping axis-aligned boxes covering every solid cell of one material class.
/// Used as the static collider. A per-cell collider for a villa is unnecessary and slow.
pub fn greedy_solid_boxes(world: &VoxelWorld) -> Vec<GreedyBox> {
    use std::collections::BTreeSet;
    let mut remaining: BTreeSet<IVec3> = world
        .blocks
        .iter()
        .filter(|(_, block)| block.material.solid())
        .map(|(cell, _)| *cell)
        .collect();
    let mut boxes = Vec::new();
    while let Some(start) = remaining.iter().next().copied() {
        let material = world.blocks[&start].material;
        let mut sx = 1;
        while remaining.contains(&IVec3::new(start.x + sx, start.y, start.z))
            && world.blocks[&IVec3::new(start.x + sx, start.y, start.z)].material == material
        {
            sx += 1;
        }
        let mut sz = 1;
        'grow_z: loop {
            for x in 0..sx {
                let cell = IVec3::new(start.x + x, start.y, start.z + sz);
                if !remaining.contains(&cell) || world.blocks[&cell].material != material {
                    break 'grow_z;
                }
            }
            sz += 1;
        }
        let mut sy = 1;
        'grow_y: loop {
            for x in 0..sx {
                for z in 0..sz {
                    let cell = IVec3::new(start.x + x, start.y + sy, start.z + z);
                    if !remaining.contains(&cell) || world.blocks[&cell].material != material {
                        break 'grow_y;
                    }
                }
            }
            sy += 1;
        }
        for y in 0..sy {
            for x in 0..sx {
                for z in 0..sz {
                    remaining.remove(&IVec3::new(start.x + x, start.y + y, start.z + z));
                }
            }
        }
        boxes.push(GreedyBox {
            min: start,
            size: IVec3::new(sx, sy, sz),
            material,
        });
    }
    boxes
}
