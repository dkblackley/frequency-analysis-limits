use serde::Deserialize;
use std::iter::Map;


pub struct encrypted_record {

    enc_id: u32,
    real_id: u32,
}

// Your shared trait
pub trait Searchable {
    fn process(&self) -> DataBase;
}


// Database 1: Users
#[derive(Debug)]
pub struct CaliMap {

    idMap: Map<i32, Location>,
    dimensions: i32, // should always be two
}

#[derive(Debug, Deserialize)]
struct Location {
    id: u32,
    longitude: f64,
    latitude: f64,
}

impl CaliMap {

    pub fn new() -> Self {

    }
}


