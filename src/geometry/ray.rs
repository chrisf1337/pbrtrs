use types::{Float, Medium, Point3f, Vector3f, INFINITY};

pub struct Ray {
    pub o: Point3f,
    pub d: Vector3f,
    pub t_max: Float,
    pub time: Float,
    pub medium: Medium,
}

impl Ray {
    pub fn new(o: Point3f, d: Vector3f, t_max: Float, time: Float, medium: Medium) -> Self {
        Ray {
            o,
            d,
            t_max,
            time,
            medium,
        }
    }

    pub fn new_infinite(o: Point3f, d: Vector3f) -> Self {
        Ray::new(o, d, INFINITY, 0.0, Medium {})
    }

    pub fn at(&self, t: Float) -> Point3f {
        self.o + self.d * t
    }
}

pub struct RayDifferential {
    pub ray: Ray,
    pub has_differentials: bool,
    pub rx_origin: Point3f,
    pub ry_origin: Point3f,
    pub rx_direction: Vector3f,
    pub ry_direction: Vector3f,
}

impl RayDifferential {
    pub fn new(ray: Ray) -> Self {
        RayDifferential {
            ray,
            has_differentials: false,
            rx_origin: Point3f::new(0.0, 0.0, 0.0),
            ry_origin: Point3f::new(0.0, 0.0, 0.0),
            rx_direction: Vector3f::new(0.0, 0.0, 0.0),
            ry_direction: Vector3f::new(0.0, 0.0, 0.0),
        }
    }

    pub fn scale_differentials(&mut self, s: Float) {
        self.rx_origin = self.ray.o + (self.rx_origin - self.ray.o) * s;
        self.ry_origin = self.ray.o + (self.ry_origin - self.ray.o) * s;
        self.rx_direction = self.ray.d + (self.rx_direction - self.ray.d) * s;
        self.ry_direction = self.ray.d + (self.ry_direction - self.ray.d) * s;
    }
}
