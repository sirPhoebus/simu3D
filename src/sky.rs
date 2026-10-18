//! Equirectangular sky. Values are radiance, not display colour: the sun is
//! several times brighter than the zenith so the shader can keep a highlight
//! without lifting the whole hemisphere to the same grey.

use crate::math3::Vec3;

pub fn sun_direction() -> Vec3 {
    Vec3::new(0.42, 0.74, -0.52).normalized()
}

pub fn radiance(dir: Vec3) -> [f32; 3] {
    let dir = dir.normalized();
    let sun = sun_direction();
    let ndot = dir.dot(sun).max(0.0);
    let disc = ndot.powf(220.0) * 18.0;
    let glow = ndot.powf(6.0) * 1.6;
    let up = dir.y;
    let sky = if up >= 0.0 {
        let t = up.clamp(0.0, 1.0);
        let horizon = Vec3::new(0.92, 0.48, 0.22);
        let zenith = Vec3::new(0.08, 0.22, 0.62);
        horizon * (1.0 - t) + zenith * t
    } else {
        Vec3::new(0.045, 0.04, 0.032) * (0.35 + 0.65 * (1.0 + up))
    };
    let warm = Vec3::new(1.0, 0.9, 0.7);
    let rgb = sky + warm * (disc + glow);
    [rgb.x, rgb.y, rgb.z]
}

/// Row y = 0 is the zenith (+Y). Each texel is rgba16f, matching the shader's
/// equirectangular lookup.
pub fn equirect_rgba16(width: u32, height: u32) -> Vec<u16> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let lon = (u - 0.5) * std::f32::consts::TAU;
            let lat = (0.5 - v) * std::f32::consts::PI;
            let dir = Vec3::new(lon.cos() * lat.cos(), lat.sin(), lon.sin() * lat.cos());
            let rgb = radiance(dir);
            out.push(f32_to_f16(rgb[0]));
            out.push(f32_to_f16(rgb[1]));
            out.push(f32_to_f16(rgb[2]));
            out.push(f32_to_f16(1.0));
        }
    }
    out
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(frac != 0);
    }
    let exp = exp - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let frac = (frac | 0x80_0000) >> (1 - exp) as u32;
        return sign | ((frac >> 13) as u16);
    }
    sign | ((exp as u16) << 10) | ((frac >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sun_is_brighter_than_the_ground_hemisphere() {
        let sun = radiance(sun_direction());
        let ground = radiance(Vec3::new(0.0, -1.0, 0.0));
        let sun_l = sun[0] + sun[1] + sun[2];
        let ground_l = ground[0] + ground[1] + ground[2];
        assert!(
            sun_l > ground_l * 8.0,
            "sun {sun_l:.2} ground {ground_l:.2}"
        );
        let pixels = equirect_rgba16(8, 4);
        assert_eq!(pixels.len(), 8 * 4 * 4);
        assert!(pixels.iter().any(|p| *p != 0));
    }
}
