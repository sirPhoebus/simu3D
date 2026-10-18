//! simu3d voxel engine.
//!
//! Gravity is constant standard gravity (default 9.80665 m/s²), the same vector for the player,
//! rockets, blast fragments, and any structure that has lost a connection to the anchored ground.
//! Airborne rigid bodies are corrected onto the closed-form free-fall trajectory after each Rapier
//! step, and a coarse step stops at a solid voxel floor instead of passing through it. Contacts
//! stay with Rapier's impulse solver.
//!
//! An explosive round does not delete voxels. Every block the fireball intersects is
//! removed from the static grid and respawned as rigid bodies: cubic-grid fragments in the
//! fireball, each piece carrying its mass share of the blast impulse, plus spin from
//! the off-centre hit. Past the fireball, the shock tears neighbour bonds and those
//! blocks fly whole. Unsupported structure is kicked by the same impulse, not dropped
//! from rest. Airborne rigid bodies then follow the same free-fall correction.

pub mod biome;
pub mod blast;
pub mod config;
pub mod level;
pub mod material;
pub mod math3;
pub mod mesh;
pub mod self_test;
pub mod sim;
pub mod sky;
pub mod split;
pub mod street;
pub mod villa;
pub mod world;
