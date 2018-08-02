use geometry::point;
use na::{Point2, Point3, Vector3};
use num::{Integer, One, Zero};
use types::{pmax, pmin, ElemType, Float, Point3f};

pub struct Bounds2<T: ElemType> {
    pub p_min: Point2<T>,
    pub p_max: Point2<T>,
}

pub struct Bounds3<T: ElemType> {
    pub p_min: Point3<T>,
    pub p_max: Point3<T>,
}

type Bounds3f = Bounds3<Float>;

impl<T: ElemType> Bounds3<T> {
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self {
        Bounds3 {
            p_min: Point3::new(pmin(p1.x, p2.x), pmin(p1.y, p2.y), pmin(p1.z, p2.z)),
            p_max: Point3::new(pmax(p1.x, p2.x), pmax(p1.y, p2.y), pmax(p1.z, p2.z)),
        }
    }

    pub fn new_inf() -> Self {
        Bounds3::new(
            Point3::new(T::max_value(), T::max_value(), T::max_value()),
            Point3::new(T::min_value(), T::min_value(), T::min_value()),
        )
    }

    pub fn new_pt(p: Point3<T>) -> Self {
        Bounds3::new(p, p)
    }

    pub fn corner(&self, corner: usize) -> Point3<T> {
        let x = (if corner & 1 > 0 {
            self.p_min
        } else {
            self.p_max
        }).x;
        let y = (if corner & 2 > 0 {
            self.p_min
        } else {
            self.p_max
        }).y;
        let z = (if corner & 4 > 0 {
            self.p_min
        } else {
            self.p_max
        }).z;
        Point3::new(x, y, z)
    }

    pub fn union_pt(&self, p: &Point3<T>) -> Bounds3<T> {
        Bounds3::new(
            Point3::new(
                pmin(self.p_min.x, p.x),
                pmin(self.p_min.y, p.y),
                pmin(self.p_min.z, p.z),
            ),
            Point3::new(
                pmax(self.p_max.x, p.x),
                pmax(self.p_max.y, p.y),
                pmax(self.p_max.z, p.z),
            ),
        )
    }

    pub fn union(&self, b: &Bounds3<T>) -> Bounds3<T> {
        Bounds3::new(
            Point3::new(
                pmin(self.p_min.x, b.p_min.x),
                pmin(self.p_min.y, b.p_min.y),
                pmin(self.p_min.z, b.p_min.z),
            ),
            Point3::new(
                pmax(self.p_max.x, b.p_max.x),
                pmax(self.p_max.y, b.p_max.y),
                pmax(self.p_max.z, b.p_max.z),
            ),
        )
    }

    pub fn intersect(&self, b: &Bounds3<T>) -> Bounds3<T> {
        Bounds3::new(
            Point3::new(
                pmax(self.p_min.x, b.p_min.x),
                pmax(self.p_min.y, b.p_min.y),
                pmax(self.p_min.z, b.p_min.z),
            ),
            Point3::new(
                pmin(self.p_max.x, b.p_max.x),
                pmin(self.p_max.y, b.p_max.y),
                pmin(self.p_max.z, b.p_max.z),
            ),
        )
    }

    pub fn overlaps(&self, b: &Bounds3<T>) -> bool {
        let x = (self.p_max.x >= b.p_min.x) && (self.p_min.x <= b.p_max.x);
        let y = (self.p_max.y >= b.p_min.y) && (self.p_min.y <= b.p_max.y);
        let z = (self.p_max.z >= b.p_min.z) && (self.p_min.z <= b.p_max.z);
        x && y && z
    }

    pub fn inside(&self, p: &Point3<T>) -> bool {
        p.x >= self.p_min.x
            && p.x <= self.p_max.x
            && p.y >= self.p_min.y
            && p.y <= self.p_max.y
            && p.z >= self.p_min.z
            && p.z <= self.p_max.z
    }

    pub fn inside_exclusive(&self, p: &Point3<T>) -> bool {
        p.x >= self.p_min.x
            && p.x < self.p_max.x
            && p.y >= self.p_min.y
            && p.y < self.p_max.y
            && p.z >= self.p_min.z
            && p.z < self.p_max.z
    }

    pub fn expand(&self, delta: T) -> Bounds3<T> {
        Bounds3::new(
            self.p_min - Vector3::new(delta, delta, delta),
            self.p_max + Vector3::new(delta, delta, delta),
        )
    }

    pub fn diagonal(&self) -> Vector3<T> {
        self.p_max - self.p_min
    }

    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        T::from_int(2) * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    pub fn volume(&self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    pub fn maximum_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    pub fn offset(&self, p: &Point3<T>) -> Vector3<T> {
        let mut o = p - self.p_min;
        if self.p_max.x > self.p_min.x {
            o.x /= self.p_max.x - self.p_min.x;
        }
        if self.p_max.y > self.p_min.y {
            o.y /= self.p_max.y - self.p_min.y;
        }
        if self.p_max.z > self.p_min.z {
            o.z /= self.p_max.z - self.p_min.z;
        }
        o
    }
}

impl<T: ElemType + Integer> IntoIterator for Bounds2<T> {
    type Item = Point2<T>;
    type IntoIter = Bounds2Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Bounds2Iter {
            p_max: self.p_max,
            x: self.p_min.x,
            y: self.p_min.y,
        }
    }
}

pub struct Bounds2Iter<T: ElemType> {
    p_max: Point2<T>,
    x: T,
    y: T,
}

impl<T: ElemType> Iterator for Bounds2Iter<T> {
    type Item = Point2<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.x >= self.p_max.x && self.y >= self.p_max.y {
            None
        } else {
            if self.x >= self.p_max.x {
                self.x = Zero::zero();
                self.y += One::one();
            } else {
                self.x += One::one();
            }
            Some(Point2::new(self.x, self.y))
        }
    }
}

pub fn bounding_sphere(bounds: &Bounds3f) -> (Point3f, Float) {
    let center = (bounds.p_min + point::to_vec3(&bounds.p_max)) / 2.0;
    (
        center,
        if bounds.inside(&center) {
            (center - bounds.p_max).norm()
        } else {
            Zero::zero()
        },
    )
}

pub fn lerp(bounds: &Bounds3f, t: &Point3f) -> Point3f {
    Point3f::new(
        ::lerp(t.x, bounds.p_min.x, bounds.p_max.x),
        ::lerp(t.y, bounds.p_min.y, bounds.p_max.y),
        ::lerp(t.z, bounds.p_min.z, bounds.p_max.z),
    )
}
