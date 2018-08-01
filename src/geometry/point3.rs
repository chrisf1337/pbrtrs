use geometry::vector3 as vec3;
use na::{Point2, Point3, Vector3};
use types::{pmax, pmin, ElemType, Float, HasNan, Point3f};

pub fn to_pt2<T: ElemType>(p: &Point3<T>) -> Point2<T> {
    assert!(!p.has_nan());
    Point2::new(p[0], p[1])
}

pub fn to_vec3<T: ElemType>(p: &Point3<T>) -> Vector3<T> {
    Vector3::new(p[0], p[1], p[2])
}

pub fn lerp(t: Float, p0: &Point3f, p1: &Point3f) -> Point3f {
    vec3::to_pt3(&((1.0 - t) * p0.coords + t * p1.coords))
}

pub fn min<T: ElemType>(p1: &Point3<T>, p2: &Point3<T>) -> Point3<T> {
    Point3::new(pmin(p1[0], p2[0]), pmin(p1[1], p2[1]), pmin(p1[2], p2[2]))
}

pub fn max<T: ElemType>(p1: &Point3<T>, p2: &Point3<T>) -> Point3<T> {
    Point3::new(pmax(p1[0], p2[0]), pmax(p1[1], p2[1]), pmax(p1[2], p2[2]))
}

pub fn floor(p: &Point3f) -> Point3f {
    Point3::new(p[0].floor(), p[1].floor(), p[2].floor())
}

pub fn ceil(p: &Point3f) -> Point3f {
    Point3::new(p[0].ceil(), p[1].ceil(), p[2].ceil())
}

pub fn abs(p: &Point3f) -> Point3f {
    Point3::new(p[0].abs(), p[1].abs(), p[2].abs())
}
