#![allow(unknown_lints)]
#![warn(clippy)]
#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
extern crate alga;
extern crate nalgebra as na;
extern crate num;

mod geometry;
pub mod parser;
mod types;

use types::Float;

pub fn lerp(t: Float, v1: Float, v2: Float) -> Float {
    (1.0 - t) * v1 + t * v2
}
