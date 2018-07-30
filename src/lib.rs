#![allow(unknown_lints)]
#![warn(clippy)]

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
extern crate nalgebra as na;

pub mod parser;
pub mod scene;
pub mod types;
