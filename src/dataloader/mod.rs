use crate::{Record, Value};

pub mod datasets;
mod error;
pub mod processing;

// Your shared trait
pub trait Searchable: Sync {
    fn get_dims(&self) -> Value;
    // fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value>;
    fn get_name(&self) -> &str;
    fn do_search(&self, lower: Record, upper: Record) -> Vec<Value>;

    /// Remember, to compute the dominating vals/prob-freq pairs we don't really need the DB, we just
    /// Need the domain, specifically the lowest and highest possible value on x/y/z/whatever.
    /// you can think of this as the 'largest dominating pair value'
    fn get_dom_pair(&self) -> (Record, Record);

    /// Returns all individual encrypted records.
    fn get_universe(&self) -> Vec<Value>;
}
