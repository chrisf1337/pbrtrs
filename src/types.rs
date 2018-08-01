use na::{Matrix3, Point2, Point3, Vector2, Vector3};
use num_traits::sign::Signed;
use std::fmt::Debug;
use std::path::PathBuf;

pub type Point2f = Point2<Float>;
pub type Point3f = Point3<Float>;
pub type Vector2f = Vector2<Float>;
pub type Vector3f = Vector3<Float>;
pub type Normal3f = Vector3<Float>;
pub type Matrix3f = Matrix3<Float>;
#[cfg(feature = "float_as_f64")]
pub type Float = f64;
#[cfg(not(feature = "float_as_f64"))]
pub type Float = f32;

#[cfg(feature = "float_as_f64")]
pub const INFINITY: Float = ::std::f64::INFINITY;
#[cfg(not(feature = "float_as_f64"))]
pub const INFINITY: Float = ::std::f32::INFINITY;

pub trait Nanable {
    fn is_nan(&self) -> bool;
}

impl Nanable for Float {
    fn is_nan(&self) -> bool {
        Float::is_nan(*self)
    }
}

impl Nanable for i64 {
    fn is_nan(&self) -> bool {
        false
    }
}

pub trait HasNan {
    fn has_nan(&self) -> bool;
}

impl<T: ElemType> HasNan for Point3<T> {
    fn has_nan(&self) -> bool {
        for i in self.iter() {
            if i.is_nan() {
                return true;
            }
        }
        false
    }
}

pub trait ElemType: Copy + PartialEq + PartialOrd + Signed + Debug + Nanable + 'static {}
impl ElemType for Float {}
impl ElemType for i64 {}

#[derive(Debug, PartialEq, Clone)]
pub enum Spectrum {
    Rgb(Float, Float, Float),
    Xyz(Float, Float, Float),
    Spectrum(Vec<(Float, Float)>),
    File(PathBuf),
    Blackbody(Float, Float),
}

pub struct Medium {}

pub fn pmin<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

pub fn pmax<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}
