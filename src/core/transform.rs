use core::{matrix::Matrix4, not_one, Float, Vector3f};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub m: Matrix4<Float>,
    m_inv: Matrix4<Float>,
}

impl Transform {
    pub fn new(mat: Matrix4<Float>, inv: Matrix4<Float>) -> Transform {
        Transform { m: mat, m_inv: inv }
    }

    pub fn new_(mat: Matrix4<Float>) -> Transform {
        Transform::new(mat, mat.inverse())
    }

    pub fn inverse(&self) -> Transform {
        Transform::new(self.m_inv, self.m)
    }

    pub fn transpose(&self) -> Transform {
        Transform::new(self.m.transpose(), self.m_inv.transpose())
    }

    pub fn is_identity(&self) -> bool {
        self.m == Matrix4::new([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn translate(delta: &Vector3f) -> Transform {
        Transform::new(
            Matrix4::new([
                [1., 0., 0., delta.x],
                [0., 1., 0., delta.y],
                [0., 0., 1., delta.z],
                [0., 0., 0., 1.],
            ]),
            Matrix4::new([
                [1., 0., 0., -delta.x],
                [0., 1., 0., -delta.y],
                [0., 0., 1., -delta.z],
                [0., 0., 0., 1.],
            ]),
        )
    }

    pub fn scale(x: Float, y: Float, z: Float) -> Transform {
        Transform::new(
            Matrix4::new([
                [x, 0., 0., 0.],
                [0., y, 0., 0.],
                [0., 0., z, 0.],
                [0., 0., 0., 1.],
            ]),
            Matrix4::new([
                [1. / x, 0., 0., 0.],
                [0., 1. / y, 0., 0.],
                [0., 0., 1. / z, 0.],
                [0., 0., z, 1.],
            ]),
        )
    }

    pub fn has_scale(&self) -> bool {
        let la2 = Vector3f::new(1., 0., 0.).transform(self).len_sq();
        let lb2 = Vector3f::new(0., 1., 0.).transform(self).len_sq();
        let lc2 = Vector3f::new(0., 0., 1.).transform(self).len_sq();
        not_one(la2) || not_one(lb2) || not_one(lc2)
    }
}
