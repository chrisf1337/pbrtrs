use na::{Point3, Vector3};
use num::Zero;
use types::{pmax, pmin, ElemType, Float, Vector3f};

pub fn min_component<T: ElemType>(v: &Vector3<T>) -> T {
    v[v.imin()]
}

pub fn max_component<T: ElemType>(v: &Vector3<T>) -> T {
    v[v.imax()]
}

pub fn max_dimension<T: ElemType>(v: &Vector3<T>) -> usize {
    if v.x > v.y {
        if v.x > v.z {
            0
        } else {
            2
        }
    } else if v.y > v.z {
        1
    } else {
        2
    }
}

pub fn min<T: ElemType>(v1: &Vector3<T>, v2: &Vector3<T>) -> Vector3<T> {
    Vector3::new(pmin(v1.x, v2.x), pmin(v1.y, v2.y), pmin(v1.z, v2.z))
}

pub fn max<T: ElemType>(v1: &Vector3<T>, v2: &Vector3<T>) -> Vector3<T> {
    Vector3::new(pmax(v1.x, v2.x), pmax(v1.y, v2.y), pmax(v1.z, v2.z))
}

pub fn permute<T: ElemType>(v: &Vector3<T>, x: usize, y: usize, z: usize) -> Vector3<T> {
    Vector3::new(v[x], v[y], v[z])
}

pub fn coordinate_system<T: ElemType>(v1: &Vector3f) -> (Vector3f, Vector3f) {
    let v2 = if v1.x.abs() > v1.y.abs() {
        Vector3::new(-v1.z, 0.0, v1.x) / Float::sqrt(v1.x * v1.x + v1.z * v1.z)
    } else {
        Vector3::new(0.0, v1.z, -v1.y) / Float::sqrt(v1.y * v1.y + v1.z * v1.z)
    };
    let v3 = v1.cross(&v2);
    (v2, v3)
}

pub fn to_pt3<T: ElemType>(v: &Vector3<T>) -> Point3<T> {
    Point3::new(v.x, v.y, v.z)
}

pub fn face_forward<T: ElemType>(v1: &Vector3<T>, v2: &Vector3<T>) -> Vector3<T> {
    if v1.dot(v2) < Zero::zero() {
        -v1
    } else {
        *v1
    }
}
