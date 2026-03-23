// Type aliases to make the code more readable
// Note: Python dictionaries can use floats as keys, but Rust HashMaps cannot due to NaN ambiguity.
// As a result we scale floats to discrete nums
pub type Coord = i64;
pub type Value = Coord;
pub type Record = Vec<Coord>;
pub type Responses = Vec<Record>;
pub type DomPair = (Record, Record);
// Using u64 here as a placeholder for your frequency type.
pub type Frequency = u64;

pub mod LAMA;
mod build;
pub mod dataloader;
pub mod plotting;
