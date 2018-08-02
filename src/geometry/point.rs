use geometry::vector as vec;
use na::{Point2, Point3, Vector3};
use types::{pmax, pmin, ElemType, Float, HasNan, Point3f};

pub fn to_pt2<T: ElemType>(p: &Point3<T>) -> Point2<T> {
    assert!(!p.has_nan());
    Point2::new(p.x, p.y)
}

pub fn to_vec3<T: ElemType>(p: &Point3<T>) -> Vector3<T> {
    Vector3::new(p.x, p.y, p.z)
}

pub fn lerp(t: Float, p0: &Point3f, p1: &Point3f) -> Point3f {
    vec::to_pt3(&((1.0 - t) * p0.coords + t * p1.coords))
}

pub fn min<T: ElemType>(p1: &Point3<T>, p2: &Point3<T>) -> Point3<T> {
    Point3::new(pmin(p1.x, p2.x), pmin(p1.y, p2.y), pmin(p1.z, p2.z))
}

pub fn max<T: ElemType>(p1: &Point3<T>, p2: &Point3<T>) -> Point3<T> {
    Point3::new(pmax(p1.x, p2.x), pmax(p1.y, p2.y), pmax(p1.z, p2.z))
}

pub fn floor(p: &Point3f) -> Point3f {
    Point3::new(p.x.floor(), p.y.floor(), p.z.floor())
}

pub fn ceil(p: &Point3f) -> Point3f {
    Point3::new(p.x.ceil(), p.y.ceil(), p.z.ceil())
}

pub fn abs<T: ElemType>(p: &Point3<T>) -> Point3<T> {
    Point3::new(p.x.abs(), p.y.abs(), p.z.abs())
}
