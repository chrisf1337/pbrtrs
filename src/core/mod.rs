pub mod boundingbox;
pub mod matrix;
pub mod normal;
pub mod point;
pub mod ray;
pub mod transform;
pub mod vector;

use alga::general::{ClosedAdd, ClosedDiv, ClosedMul, ClosedSub};
use core::{
    matrix::{Matrix3, Matrix4},
    normal::Normal3,
    point::{Point2, Point3},
    vector::{Vector2, Vector3},
};
use num::{cast::AsPrimitive, Signed};
use std::fmt::Debug;
use std::path::PathBuf;

pub fn lerp(t: Float, v1: Float, v2: Float) -> Float {
    (1.0 - t) * v1 + t * v2
}

pub type Point2f = Point2<Float>;
pub type Point3f = Point3<Float>;
pub type Vector2f = Vector2<Float>;
pub type Vector3f = Vector3<Float>;
pub type Normal3f = Normal3<Float>;
pub type Matrix3f = Matrix3<Float>;
pub type Matrix4f = Matrix4<Float>;
#[cfg(feature = "float_as_f64")]
pub type Float = f64;
#[cfg(not(feature = "float_as_f64"))]
pub type Float = f32;

#[cfg(feature = "float_as_f64")]
pub const INFINITY: Float = ::std::f64::INFINITY;
#[cfg(not(feature = "float_as_f64"))]
pub const INFINITY: Float = ::std::f32::INFINITY;

pub trait ElemType:
    Copy
    + PartialEq
    + PartialOrd
    + Signed
    + Debug
    + ClosedAdd
    + ClosedSub
    + ClosedMul
    + ClosedDiv
    + AsPrimitive<Float>
    + AsPrimitive<i64>
    + 'static
{
    fn is_nan(&self) -> bool;
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn from_int(i: isize) -> Self;
}

impl ElemType for Float {
    fn is_nan(&self) -> bool {
        Float::is_nan(*self)
    }

    fn max_value() -> Float {
        INFINITY
    }

    fn min_value() -> Float {
        -INFINITY
    }

    fn from_int(i: isize) -> Float {
        i as Float
    }
}

impl ElemType for i64 {
    fn is_nan(&self) -> bool {
        false
    }

    fn max_value() -> i64 {
        i64::max_value()
    }

    fn min_value() -> i64 {
        i64::min_value()
    }

    fn from_int(i: isize) -> i64 {
        i as i64
    }
}

pub trait ToFloat {
    fn to_float(self) -> Float;
}

impl ToFloat for Float {
    fn to_float(self) -> Float {
        self
    }
}

impl ToFloat for i64 {
    fn to_float(self) -> Float {
        self as Float
    }
}

pub trait HasNan {
    fn has_nan(&self) -> bool;
}

impl<T: ElemType> HasNan for Point3<T> {
    fn has_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Spectrum {
    Rgb(Float, Float, Float),
    Xyz(Float, Float, Float),
    Spectrum(Vec<(Float, Float)>),
    File(PathBuf),
    Blackbody(Float, Float),
}

pub struct Medium {}

pub fn pmin<T: PartialOrd + Copy>(elems: &[T]) -> T {
    let mut min = &elems[0];
    for e in &elems[1..] {
        if e < min {
            min = e;
        }
    }
    *min
}

pub fn pmax<T: PartialOrd + Copy>(elems: &[T]) -> T {
    let mut max = &elems[0];
    for e in &elems[1..] {
        if e > max {
            max = e;
        }
    }
    *max
}

pub fn not_one(x: Float) -> bool {
    x < 0.999 || x > 1.001
}
