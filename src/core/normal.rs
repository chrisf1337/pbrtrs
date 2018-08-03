use core::{pmax, pmin, vector::Vector3, ElemType, Float};
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal3<T: ElemType> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: ElemType> Index<usize> for Normal3<T> {
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

impl<T: ElemType> Normal3<T> {
    pub fn new(x: T, y: T, z: T) -> Normal3<T> {
        Normal3 { x, y, z }
    }

    pub fn len_sq(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn len(&self) -> Float {
        let f: Float = self.len_sq().as_();
        f.sqrt()
    }

    pub fn dot(&self, v: &Normal3<T>) -> T {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn cross(&self, v: &Normal3<T>) -> Normal3<T> {
        Normal3::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }

    pub fn permute(&self, x: usize, y: usize, z: usize) -> Normal3<T> {
        Normal3::new(self[x], self[y], self[z])
    }

    pub fn face_forward(&self, v2: &Normal3<T>) -> Normal3<T> {
        if self.dot(v2) < T::from_int(0) {
            -self
        } else {
            *self
        }
    }

    pub fn min(&self, v: &Normal3<T>) -> Normal3<T> {
        Normal3::new(
            pmin(&[self.x, v.x]),
            pmin(&[self.y, v.y]),
            pmin(&[self.z, v.z]),
        )
    }

    pub fn max(&self, v: &Normal3<T>) -> Normal3<T> {
        Normal3::new(
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
}

impl<T: ElemType> Neg for Normal3<T> {
    type Output = Normal3<T>;

    fn neg(self) -> Self::Output {
        Normal3::new(-self.x, -self.y, -self.z)
    }
}

impl<'a, T: ElemType> Neg for &'a Normal3<T> {
    type Output = Normal3<T>;

    fn neg(self) -> Self::Output {
        Normal3::new(-self.x, -self.y, -self.z)
    }
}

impl<T: ElemType> Add<Normal3<T>> for Normal3<T> {
    type Output = Normal3<T>;

    fn add(self, other: Normal3<T>) -> Self::Output {
        Normal3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T: ElemType> Sub<Normal3<T>> for Normal3<T> {
    type Output = Normal3<T>;

    fn sub(self, other: Normal3<T>) -> Self::Output {
        Normal3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: ElemType> Mul<T> for Normal3<T> {
    type Output = Normal3<T>;

    fn mul(self, other: T) -> Self::Output {
        Normal3::new(other * self.x, other * self.y, other * self.z)
    }
}

impl<'a, T: ElemType> Mul<T> for &'a Normal3<T> {
    type Output = Normal3<T>;

    fn mul(self, other: T) -> Self::Output {
        Normal3::new(other * self.x, other * self.y, other * self.z)
    }
}

impl<T: ElemType> Div<T> for Normal3<T> {
    type Output = Normal3<T>;

    fn div(self, rhs: T) -> Self::Output {
        Normal3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<T: ElemType> From<Vector3<T>> for Normal3<T> {
    fn from(v: Vector3<T>) -> Normal3<T> {
        Normal3::new(v.x, v.y, v.z)
    }
}
