use core::point::Point3;
use core::{pmax, pmin, transform::Transform, ElemType, Float, Vector3f};
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2<T: Clone> {
    pub x: T,
    pub y: T,
}

impl<T: Clone> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 2);
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => unimplemented!(),
        }
    }
}

impl<T: Clone> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: ElemType> Div<T> for Vector2<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        assert!(rhs != T::from_int(0), "divide by zero");
        let recip = rhs;
        Vector2::new(self.x * recip, self.y * recip)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3<T: Clone> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Clone> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 3);
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => unimplemented!(),
        }
    }
}

impl<T: ElemType> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    pub fn len_sq(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn len(&self) -> Float {
        let f: Float = self.len_sq().as_();
        f.sqrt()
    }

    pub fn dot(&self, v: &Self) -> T {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn cross(&self, v: &Self) -> Self {
        Self::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }

    pub fn permute(&self, x: usize, y: usize, z: usize) -> Self {
        Self::new(self[x], self[y], self[z])
    }

    pub fn face_forward(&self, v2: &Self) -> Self {
        if self.dot(v2) < T::from_int(0) {
            -self
        } else {
            *self
        }
    }

    pub fn min(&self, v: &Self) -> Self {
        Self::new(
            pmin(&[self.x, v.x]),
            pmin(&[self.y, v.y]),
            pmin(&[self.z, v.z]),
        )
    }

    pub fn max(&self, v: &Self) -> Self {
        Self::new(
            pmax(&[self.x, v.x]),
            pmax(&[self.y, v.y]),
            pmax(&[self.z, v.z]),
        )
    }

    pub fn min_component(&self) -> T {
        pmin(&[self.x, self.y, self.z])
    }

    pub fn max_component(&self) -> T {
        pmax(&[self.x, self.y, self.z])
    }

    pub fn max_dimension(&self) -> usize {
        if self.x > self.y {
            if self.x > self.z {
                0
            } else {
                2
            }
        } else if self.y > self.z {
            1
        } else {
            2
        }
    }

    pub fn transform(&self, t: &Transform) -> Vector3f {
        let x: Float = self.x.as_();
        let y: Float = self.y.as_();
        let z: Float = self.z.as_();
        Vector3::new(
            t.m[0][0] * x + t.m[0][1] * y + t.m[0][2] * z,
            t.m[1][0] * x + t.m[1][1] * y + t.m[1][2] * z,
            t.m[2][0] * x + t.m[2][1] * y + t.m[2][2] * z,
        )
    }
}

impl<T: ElemType> Neg for Vector3<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl<'a, T: ElemType> Neg for &'a Vector3<T> {
    type Output = Vector3<T>;

    fn neg(self) -> Self::Output {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl<T: ElemType> Add<Vector3<T>> for Vector3<T> {
    type Output = Self;

    fn add(self, other: Vector3<T>) -> Self::Output {
        Vector3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T: ElemType> Sub<Vector3<T>> for Vector3<T> {
    type Output = Self;

    fn sub(self, other: Vector3<T>) -> Self::Output {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: ElemType> Mul<T> for Vector3<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Vector3::new(other * self.x, other * self.y, other * self.z)
    }
}

impl<'a, T: ElemType> Mul<T> for &'a Vector3<T> {
    type Output = Vector3<T>;

    fn mul(self, other: T) -> Self::Output {
        Vector3::new(other * self.x, other * self.y, other * self.z)
    }
}

impl<T: ElemType> Div<T> for Vector3<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Vector3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<T: ElemType> From<Point3<T>> for Vector3<T> {
    fn from(p: Point3<T>) -> Self {
        Self::new(p.x, p.y, p.z)
    }
}

pub fn coordinate_system<T: ElemType>(v1: &Vector3f) -> (Vector3f, Vector3f) {
    let v2 = if v1.x.abs() > v1.y.abs() {
        Vector3::new(-v1.z, 0.0, v1.x) / Float::sqrt(v1.x * v1.x + v1.z * v1.z)
    } else {
        Vector3::new(0.0, v1.z, -v1.y) / Float::sqrt(v1.y * v1.y + v1.z * v1.z)
    };
    let v3 = v1.cross(&v2);
    (v2, v3)
}
