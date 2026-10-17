//! Small 3-vectors. Kept local so the simulation does not depend on a particular glam version.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1.0e-8 {
            Self::ZERO
        } else {
            self * (1.0 / len)
        }
    }

    pub fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    pub fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new(
            self.x.clamp(lo.x, hi.x),
            self.y.clamp(lo.y, hi.y),
            self.z.clamp(lo.z, hi.z),
        )
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl IVec3 {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
}

impl std::ops::Add for IVec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

/// Unit quaternion rotating `from` onto `to`. xyzw.
pub fn quat_from_to(from: Vec3, to: Vec3) -> [f32; 4] {
    let a = from.normalized();
    let b = to.normalized();
    if a.length() < 1.0e-6 || b.length() < 1.0e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let c = a.cross(b);
    let d = a.dot(b);
    if d < -0.9999 {
        let axis = if a.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0).cross(a)
        } else {
            Vec3::Y.cross(a)
        }
        .normalized();
        return [axis.x, axis.y, axis.z, 0.0];
    }
    let w = 1.0 + d;
    let len = (c.dot(c) + w * w).sqrt();
    [c.x / len, c.y / len, c.z / len, w / len]
}

/// Rotate `v` by a unit quaternion xyzw.
pub fn quat_rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    let u = Vec3::new(q[0], q[1], q[2]);
    let s = q[3];
    let uv = u.cross(v);
    let uuv = u.cross(uv);
    v + uv * (2.0 * s) + uuv * 2.0
}

pub const NEIGHBOURS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];
