use na::{Matrix3, Point2, Point3, Vector2, Vector3};
use std::path::PathBuf;

pub type Point2f = Point2<f64>;
pub type Point3f = Point3<f64>;
pub type Vector2f = Vector2<f64>;
pub type Vector3f = Vector3<f64>;
pub type Normal3f = Vector3<f64>;
pub type Matrix3f = Matrix3<f64>;

#[derive(Debug, PartialEq, Clone)]
pub enum Spectrum {
    Rgb(f64, f64, f64),
    Xyz(f64, f64, f64),
    Spectrum(Vec<(f64, f64)>),
    File(PathBuf),
    Blackbody(f64, f64),
}
