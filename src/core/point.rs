use core::vector::Vector3;
use core::{pmax, pmin, ElemType, Float, HasNan, Point3f};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2<T: Clone> {
    pub x: T,
    pub y: T,
}

impl<T: ElemType> Point2<T> {
    pub fn new(x: T, y: T) -> Point2<T> {
        Point2 { x, y }
    }

    pub fn abs(&self) -> Point2<T> {
        Point2::new(self.x.abs(), self.y.abs())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3<T: Clone> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: ElemType> Point3<T> {
    pub fn new(x: T, y: T, z: T) -> Point3<T> {
        Point3 { x, y, z }
    }

    pub fn abs(&self) -> Point3<T> {
        Point3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn min(&self, p: &Point3<T>) -> Point3<T> {
        Point3::new(
            pmin(&[self.x, p.x]),
            pmin(&[self.y, p.y]),
            pmin(&[self.z, p.z]),
        )
    }

    pub fn max(&self, p: &Point3<T>) -> Point3<T> {
        Point3::new(
            pmax(&[self.x, p.x]),
            pmax(&[self.y, p.y]),
            pmax(&[self.z, p.z]),
        )
    }
}

impl<T: ElemType> From<Vector3<T>> for Point3<T> {
    fn from(v: Vector3<T>) -> Point3<T> {
        Point3::new(v.x, v.y, v.z)
    }
}

impl<T: ElemType> From<Point3<T>> for Point2<T> {
    fn from(p: Point3<T>) -> Point2<T> {
        assert!(!p.has_nan());
        Point2::new(p.x, p.y)
    }
}

impl<T: ElemType> Add<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn add(self, other: Vector3<T>) -> Self::Output {
        Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T: ElemType> Sub<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn sub(self, other: Vector3<T>) -> Self::Output {
        Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: ElemType> Sub<Point3<T>> for Point3<T> {
    type Output = Vector3<T>;

    fn sub(self, other: Point3<T>) -> Self::Output {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<'a, T: ElemType> Sub<Point3<T>> for &'a Point3<T> {
    type Output = Vector3<T>;

    fn sub(self, other: Point3<T>) -> Self::Output {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: ElemType> Mul<T> for Point3<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Point3::new(other * self.x, other * self.y, other * self.z)
    }
}

impl<'a, T: ElemType> Mul<T> for &'a Point3<T> {
    type Output = Point3<T>;

    fn mul(self, other: T) -> Self::Output {
        Point3::new(other * self.x, other * self.y, other * self.z)
    }
}

impl<T: ElemType> Div<T> for Point3<T> {
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        Point3::new(self.x / other, self.y / other, self.z / other)
    }
}

pub fn lerp(t: Float, p0: &Point3f, p1: &Point3f) -> Point3f {
    p0 * (1.0 - t) + Vector3::from(p1 * t)
}

pub fn floor(p: &Point3f) -> Point3f {
    Point3::new(p.x.floor(), p.y.floor(), p.z.floor())
}

pub fn ceil(p: &Point3f) -> Point3f {
    Point3::new(p.x.ceil(), p.y.ceil(), p.z.ceil())
}
