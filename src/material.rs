//! House materials. Density is the physical density used for fragment mass. Alpha below 1 is drawn in the transparent pass.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Material {
    Plaster,
    Concrete,
    Glass,
    Wood,
    WoodDark,
    Metal,
    Stone,
    Grass,
    Water,
    Fabric,
    Linen,
    Roof,
    Tile,
    Soil,
}

impl Material {
    pub fn solid(self) -> bool {
        !matches!(self, Material::Water)
    }

    pub fn transparent(self) -> bool {
        matches!(self, Material::Glass | Material::Water)
    }

    /// kg/m³. Ordinary published bulk densities, not tuned for gameplay.
    pub fn density(self) -> f32 {
        match self {
            Material::Plaster => 1400.0,
            Material::Concrete => 2400.0,
            Material::Glass => 2500.0,
            Material::Wood => 620.0,
            Material::WoodDark => 720.0,
            Material::Metal => 7850.0,
            Material::Stone => 2600.0,
            Material::Grass => 1400.0,
            Material::Water => 1000.0,
            Material::Fabric => 280.0,
            Material::Linen => 300.0,
            Material::Roof => 1200.0,
            Material::Tile => 2000.0,
            Material::Soil => 1600.0,
        }
    }

    /// Impulse (N·s) one face-bond can take before the neighbour is torn loose.
    pub fn bond_impulse(self) -> f32 {
        match self {
            Material::Glass => 6.0,
            Material::Fabric | Material::Linen => 4.0,
            Material::Grass => 8.0,
            Material::Soil => 14.0,
            Material::Wood => 22.0,
            Material::WoodDark => 26.0,
            Material::Plaster => 28.0,
            Material::Roof => 36.0,
            Material::Water => 0.0,
            Material::Tile => 48.0,
            Material::Concrete => 55.0,
            Material::Stone => 70.0,
            Material::Metal => 260.0,
        }
    }

    /// Fraction of the incoming impulse passed through this block into a neighbour.
    pub fn shock_gain(self) -> f32 {
        match self {
            Material::Glass => 0.08,
            Material::Fabric | Material::Linen => 0.05,
            Material::Grass | Material::Soil => 0.2,
            Material::Wood | Material::WoodDark => 0.32,
            Material::Plaster | Material::Roof => 0.4,
            Material::Tile | Material::Concrete => 0.55,
            Material::Stone => 0.6,
            Material::Metal => 0.72,
            Material::Water => 0.0,
        }
    }

    pub fn color(self) -> [f32; 4] {
        match self {
            Material::Plaster => [0.93, 0.91, 0.87, 1.0],
            Material::Concrete => [0.55, 0.54, 0.51, 1.0],
            Material::Glass => [0.62, 0.78, 0.84, 0.30],
            Material::Wood => [0.64, 0.44, 0.26, 1.0],
            Material::WoodDark => [0.28, 0.16, 0.09, 1.0],
            Material::Metal => [0.16, 0.17, 0.18, 1.0],
            Material::Stone => [0.66, 0.64, 0.60, 1.0],
            Material::Grass => [0.34, 0.50, 0.24, 1.0],
            Material::Water => [0.12, 0.42, 0.58, 0.48],
            Material::Fabric => [0.32, 0.36, 0.42, 1.0],
            Material::Linen => [0.90, 0.88, 0.84, 1.0],
            Material::Roof => [0.22, 0.23, 0.24, 1.0],
            Material::Tile => [0.86, 0.88, 0.90, 1.0],
            Material::Soil => [0.40, 0.30, 0.20, 1.0],
        }
    }
}
