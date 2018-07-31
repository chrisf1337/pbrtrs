#![allow(unknown_lints)]
#![warn(clippy)]
#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
extern crate nalgebra as na;
extern crate num_traits;

mod geometry;
pub mod parser;
mod types;
