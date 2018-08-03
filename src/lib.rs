#![allow(unknown_lints)]
#![warn(clippy)]
#![allow(too_many_arguments)]
#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
extern crate alga;
extern crate num;

mod core;
pub mod parser;

pub use core::matrix::Matrix4;
