#[derive(Debug, PartialEq)]
pub struct Point2f {
    x: f64,
    y: f64,
}

impl Point2f {
    pub fn new(x: f64, y: f64) -> Self {
        Point2f { x, y }
    }
}

pub struct Vector2f {
    x: f64,
    y: f64,
}

pub struct Point3f {
    x: f64,
    y: f64,
    z: f64,
}

pub struct Vector3f {
    x: f64,
    y: f64,
    z: f64,
}

pub struct Normal3f {
    x: f64,
    y: f64,
    z: f64,
}

pub struct Spectrum {}
