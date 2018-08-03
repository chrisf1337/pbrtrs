#[derive(Debug, PartialEq)]
pub struct Matrix3<T> {
    pub mat: [[T; 3]; 3],
}

#[derive(Debug, PartialEq)]
pub struct Matrix4<T> {
    pub mat: [[T; 4]; 4],
}

impl<T> Matrix3<T> {
    pub fn new(a00: T, a01: T, a02: T, a10: T, a11: T, a12: T, a20: T, a21: T, a22: T) -> Self {
        Matrix3 {
            mat: [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]],
        }
    }
}
