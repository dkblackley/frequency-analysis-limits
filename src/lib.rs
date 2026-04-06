// Type aliases to make the code more readable
// Note: Python dictionaries can use floats as keys, but Rust HashMaps cannot due to NaN ambiguity.
// As a result we scale floats to discrete nums
pub type Coord = i64;
pub type Value = Coord;
pub type Record = Vec<Coord>;
pub type Responses = Vec<Record>;
pub type DomPair = (Record, Record);
// A Probability between 0-1. When used for a response: What's the probability of seeing this response?
pub type Probability = f64;
// Stores a discrete count used in imperfect knowledge: "count how many I actually observed". Should inevitably end up as a probability.
pub type Frequency = u64;

pub mod LAMA;
mod build;
pub mod dataloader;
pub mod plotting;
