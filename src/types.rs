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

#[derive(Debug, PartialEq)]
pub struct Vector2f {
    x: f64,
    y: f64,
}

impl Vector2f {
    pub fn new(x: f64, y: f64) -> Self {
        Vector2f { x, y }
    }
}

#[derive(Debug, PartialEq)]
pub struct Point3f {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3f {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3f { x, y, z }
    }
}

#[derive(Debug, PartialEq)]
pub struct Vector3f {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3f {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3f { x, y, z }
    }
}

#[derive(Debug, PartialEq)]
pub struct Normal3f {
    x: f64,
    y: f64,
    z: f64,
}

impl Normal3f {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Normal3f { x, y, z }
    }
}

#[derive(Debug, PartialEq)]
pub struct Spectrum {}
