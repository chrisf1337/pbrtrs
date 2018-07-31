extern crate pbrtrs;

use pbrtrs::parser::Parser;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = Path::new(&args[1]);
    let mut file = File::open(filename).expect(&format!("opening file at {}", args[1]));
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("reading file");
    let _ = Parser::parse(&contents);
}
