use dataloader::datasets::Searchable;
use frequency_analysis_limits::dataloader;
use frequency_analysis_limits::dataloader::processing::Value;
use std::collections::HashMap;

#[derive(Debug)]
pub struct testDB {
    //Map of 'node_id' to the two lat and longs (multiplied by 100 and cast to u64)
    idMap: HashMap<Value, Vec<Value>>,
    dimensions: Value, // should always be two
    name: String,
}

impl Default for testDB {
    fn default() -> Self {
        Self::new()
    }
}

impl testDB {
    pub fn new() -> Self {
        let mut id_map = HashMap::new();
        for i in 0..40 {
            id_map.insert(i as Value, vec![(i) as Value, (i / 2 as Value)]);
        }

        Self {
            idMap: id_map,
            dimensions: 2,
            name: "testDB".to_string(),
        }
    }
}

impl Searchable for testDB {
    type Key = Value;
    type Value = Vec<Value>;

    fn get_dims(&self) -> Value {
        self.dimensions
    }

    fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value> {
        &self.idMap
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
