extern crate pbrtrs;

use pbrtrs::parser::Parser;
use pbrtrs::Matrix4;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() {
    // let args: Vec<String> = env::args().collect();
    // let filename = Path::new(&args[1]);
    // let mut file = File::open(filename).expect(&format!("opening file at {}", args[1]));
    // let mut contents = String::new();
    // file.read_to_string(&mut contents).expect("reading file");
    // let _ = Parser::parse(&contents);

    let mat = Matrix4::new([[7, 1, 9, 5], [4, 5, 2, 4], [3, 7, 8, 8], [1, 9, 8, 9]]);
    println!("{:#?}", mat.inverse());
}
