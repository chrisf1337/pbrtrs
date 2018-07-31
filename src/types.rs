use na::{Matrix3, Point2, Point3, Vector2, Vector3};
use num_traits::sign::Signed;
use std::fmt::Debug;
use std::path::PathBuf;

pub type Point2f = Point2<f64>;
pub type Point3f = Point3<f64>;
pub type Vector2f = Vector2<f64>;
pub type Vector3f = Vector3<f64>;
pub type Normal3f = Vector3<f64>;
pub type Matrix3f = Matrix3<f64>;

pub trait Nanable {
    fn is_nan(&self) -> bool;
}

impl Nanable for f64 {
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
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

#[derive(Debug, PartialEq, Clone)]
pub enum Spectrum {
    Rgb(f64, f64, f64),
    Xyz(f64, f64, f64),
    Spectrum(Vec<(f64, f64)>),
    File(PathBuf),
    Blackbody(f64, f64),
}

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
