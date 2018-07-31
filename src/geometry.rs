use na::{Point2, Point3, Vector3};
use types::*;

pub fn min_component<T: ElemType>(v: &Vector3<T>) -> T {
    v[v.imin()]
}

pub fn max_component<T: ElemType>(v: &Vector3<T>) -> T {
    v[v.imax()]
}

pub fn max_dimension<T: ElemType>(v: &Vector3<T>) -> usize {
    if v[0] > v[1] {
        if v[0] > v[3] {
            0
        } else {
            2
        }
    } else if v[1] > v[2] {
        1
    } else {
        2
    }
}

pub fn min<T: ElemType>(v1: &Vector3<T>, v2: &Vector3<T>) -> Vector3<T> {
    Vector3::new(pmin(v1[0], v2[0]), pmin(v1[1], v2[1]), pmin(v1[2], v2[2]))
}

pub fn max<T: ElemType>(v1: &Vector3<T>, v2: &Vector3<T>) -> Vector3<T> {
    Vector3::new(pmax(v1[0], v2[0]), pmax(v1[1], v2[1]), pmax(v1[2], v2[2]))
}

pub fn permute<T: ElemType>(v: &Vector3<T>, x: usize, y: usize, z: usize) -> Vector3<T> {
    Vector3::new(v[x], v[y], v[z])
}

pub fn coordinate_system<T: ElemType>(v1: &Vector3f) -> (Vector3f, Vector3f) {
    let v2 = if v1[0].abs() > v1[1].abs() {
        Vector3::new(-v1[2], 0.0, v1[0]) / f64::sqrt(v1[0] * v1[0] + v1[2] * v1[2])
    } else {
        Vector3::new(0.0, v1[2], -v1[1]) / f64::sqrt(v1[1] * v1[1] + v1[2] * v1[2])
    };
    let v3 = v1.cross(&v2);
    (v2, v3)
}

pub fn pt3_to_pt2<T: ElemType>(p: &Point3<T>) -> Point2<T> {
    assert!(!p.has_nan());
    Point2::new(p[0], p[1])
}
