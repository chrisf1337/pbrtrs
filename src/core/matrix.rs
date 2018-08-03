use core::{ElemType, Float, Matrix4f};
use std::ops::{Add, Index, Mul};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Matrix3<T> {
    pub mat: [[T; 3]; 3],
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Matrix4<T> {
    pub mat: [[T; 4]; 4],
}

impl<T: Copy> Matrix3<T> {
    pub fn new(mat: [[T; 3]; 3]) -> Matrix3<T> {
        Matrix3 { mat }
    }
}

impl<T: ElemType> Matrix4<T> {
    pub fn new(mat: [[T; 4]; 4]) -> Matrix4<T> {
        Matrix4 { mat }
    }

    pub fn transpose(&self) -> Matrix4<T> {
        let m = self.mat;
        Matrix4::new([
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ])
    }

    pub fn inverse(&self) -> Matrix4f {
        let mut indxc = [0; 4];
        let mut indxr = [0; 4];
        let mut ipiv = [0; 4];
        let mut minv = to_mat4f(self).mat;
        for i in 0..4 {
            let mut irow = 0;
            let mut icol = 0;
            let mut big: Float = 0.0;
            // Choose pivot
            for j in 0..4 {
                if ipiv[j] != 1 {
                    for k in 0..4 {
                        if ipiv[k] == 0 {
                            if minv[j][k].abs() >= big {
                                big = Float::abs(minv[j][k]);
                                irow = j;
                                icol = k;
                            }
                        } else if ipiv[k] > 1 {
                            panic!("Singular matrix in Matrix4::inverse");
                        }
                    }
                }
            }
            ipiv[icol] += 1;
            // Swap rows _irow_ and _icol_ for pivot
            if irow != icol {
                for k in 0..4usize {
                    mat4swap(&mut minv, (irow, k), (icol, k));
                }
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if minv[icol][icol] == 0.0 {
                panic!("Singular matrix in Matrix::inverse");
            }

            // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            let pivinv: Float = 1. / minv[icol][icol];
            minv[icol][icol] = 1.;
            for j in 0..4 {
                minv[icol][j] *= pivinv;
            }

            // Subtract this row from others to zero out their columns
            for j in 0..4 {
                if j != icol {
                    let save = minv[j][icol];
                    minv[j][icol] = 0.;
                    for k in 0..4 {
                        minv[j][k] -= minv[icol][k] * save
                    }
                }
            }
        }
        // Swap columns to reflect permutation
        for j in (0..4).rev() {
            if indxr[j] != indxc[j] {
                for k in 0..4 {
                    mat4swap(&mut minv, (k, indxr[j]), (k, indxc[j]));
                }
            }
        }
        Matrix4::new(minv)
    }
}

pub fn mat4swap<T: Copy>(
    mat: &mut [[T; 4]; 4],
    (r0, c0): (usize, usize),
    (r1, c1): (usize, usize),
) {
    let tmp = mat[r0][c0];
    mat[r0][c0] = mat[r1][c1];
    mat[r1][c1] = tmp;
}

pub fn to_mat4f<T: ElemType>(mat: &Matrix4<T>) -> Matrix4f {
    let mut new_mat: [[Float; 4]; 4] = [[0.; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            new_mat[r][c] = mat.mat[r][c].as_();
        }
    }
    Matrix4f::new(new_mat)
}

impl<T: ElemType> Add for Matrix4<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut mat = [[T::from_int(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                mat[r][c] = self.mat[r][c] + rhs.mat[r][c]
            }
        }
        Matrix4::new(mat)
    }
}

impl<T: ElemType> Mul for Matrix4<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut mat = [[T::from_int(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                mat[r][c] = (0..4usize)
                    .map(|i| self.mat[r][i] * rhs.mat[i][c])
                    .fold(T::from_int(0), Add::add)
            }
        }
        Matrix4::new(mat)
    }
}

impl<T: ElemType> Index<usize> for Matrix4<T> {
    type Output = [T; 4];

    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 4);
        &self.mat[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul() {
        let m1 = Matrix4::new([
            [-1, 0, 4, -1],
            [5, -1, -8, -9],
            [7, -4, 5, -4],
            [-2, -10, 1, 4],
        ]);
        let m2 = Matrix4::new([
            [-10, 10, -8, -3],
            [2, 6, 0, -9],
            [7, -1, 10, 9],
            [-8, -2, -4, 5],
        ]);

        assert_eq!(
            m1 * m2,
            Matrix4::new([
                [46, -12, 52, 34],
                [-36, 70, -84, -123],
                [-11, 49, 10, 40],
                [-25, -89, 10, 125]
            ])
        );
    }

    #[test]
    fn test_inverse() {
        let mat = Matrix4::new([
            [4., 0., 0., 0.],
            [0., 0., 2., 0.],
            [0., 1., 2., 0.],
            [1., 0., 0., 1.],
        ]);
        assert_eq!(
            mat.inverse(),
            Matrix4::new([
                [0.25, 0., 0., 0.],
                [0., -1., 1., 0.],
                [0., 0.5, 0., 0.],
                [-0.25, 0., 0., 1.]
            ])
        );
    }
}
