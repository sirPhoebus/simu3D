//! Blast impulse. This is not a voxel delete.
//!
//! `j_per_area` is the specific impulse (N·s/m²) on a surface at the charge.
//! A block of face area A at distance d receives J = j_per_area * A * falloff(d).
//! Its centre-of-mass velocity is J / m, with m = density * volume.
//! The same J applied on the near face produces torque τ = r × J and spin I⁻¹τ.
//!
//! Falloff is (1 − d/R)² inside the fireball, with a floor so the surface of the
//! hole still throws, then an exponential tail past R so the shock can break the
//! next course of blocks. Neighbours receive J * material.shock_gain(). A bond
//! breaks when that exceeds material.bond_impulse().

use crate::math3::Vec3;

pub fn face_impulse(distance: f32, radius: f32, j_per_area: f32, face_area: f32) -> f32 {
    let radius = radius.max(0.05);
    let x = (distance / radius).max(0.0);
    let falloff = if x <= 1.0 {
        (1.0 - x).powi(2) * 0.60 + 0.40
    } else {
        0.40 * (-(x - 1.0) * 2.6).exp()
    };
    (j_per_area * face_area * falloff).max(0.0)
}

pub fn ejection_velocity(dir: Vec3, impulse: f32, mass: f32) -> Vec3 {
    let speed = (impulse / mass.max(1.0e-6)).clamp(0.0, 42.0);
    dir * speed
}

/// Spin from an impulse applied on the face toward the charge, box axes aligned.
pub fn ejection_spin(dir: Vec3, impulse: f32, mass: f32, size: Vec3) -> Vec3 {
    let reach = size.x.max(size.y).max(size.z) * 0.5;
    let r = dir * -reach * 0.35 + tangent_offset(dir) * reach;
    let torque = r.cross(dir * impulse);
    let m = mass.max(1.0e-6);
    let ixx = m / 12.0 * (size.y * size.y + size.z * size.z);
    let iyy = m / 12.0 * (size.x * size.x + size.z * size.z);
    let izz = m / 12.0 * (size.x * size.x + size.y * size.y);
    let spin = Vec3::new(
        torque.x / ixx.max(1.0e-5),
        torque.y / iyy.max(1.0e-5),
        torque.z / izz.max(1.0e-5),
    );
    let mag = spin.length();
    if mag > 12.0 {
        spin * (12.0 / mag)
    } else {
        spin
    }
}

fn tangent_offset(dir: Vec3) -> Vec3 {
    let axis = if dir.y.abs() < 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    axis.cross(dir).normalized()
}

pub fn radial_dir(from: Vec3, at: Vec3, fallback: Vec3) -> Vec3 {
    let offset = at - from;
    let d = offset.length();
    if d < 1.0e-4 {
        fallback
    } else {
        offset * (1.0 / d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulse_uses_the_fragments_actual_mass() {
        let velocity = ejection_velocity(Vec3::Y, 0.1, 0.01);
        assert!((velocity.y - 10.0).abs() < 1.0e-5);
    }
}
