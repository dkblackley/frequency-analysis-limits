use crate::dataloader::error::DataLoadingError;
use csv::ReaderBuilder;
use log::trace;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

// Your shared trait
pub trait Searchable {
    type Key;
    type Value;

    fn get_dims(&self) -> i64;
    fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value>;
    fn get_name(&self) -> &str;
}

// Database 1: Users
#[derive(Debug)]
pub struct CaliMap {
    //Map of 'node_id' to the two lat and longs (multiplied by 100 and cast to u64)
    idMap: HashMap<u32, Vec<(u64, u64)>>,
    dimensions: i64, // should always be two
    name: String,
}

#[derive(Debug, Deserialize)]
struct Location {
    id: u32,
    longitude: f64,
    latitude: f64,
}

impl CaliMap {
    pub fn new(file_path: &str) -> Result<Self, DataLoadingError> {
        let file = File::open(file_path)?;

        // Configure the CSV reader
        let mut rdr = ReaderBuilder::new()
            .has_headers(false) // Critical: tells the parser not to skip the first row
            .delimiter(b'\t') // Set the delimiter to a tab (use b' ' if it's actually space-separated)
            .from_reader(file);

        let mut locations = HashMap::new();
        // Iterate over the rows and deserialize them into our struct
        for result in rdr.deserialize() {
            // The `?` operator unpackages the Result, returning an error if parsing fails
            let record: Location = result.map_err(|csv_err| DataLoadingError::Parsing {
                file_path: file_path.to_string(),
                error: csv_err.to_string(),
            })?;

            // You now have a strongly typed struct
            trace!(
                "ID: {}, Lat: {}, Lon: {}",
                record.id, record.latitude, record.longitude
            );
            let scaled_lat = (record.latitude * 100.0) as u64;
            let scaled_lon = (record.longitude * 100.0) as u64;

            locations.insert(record.id, vec![(scaled_lat, scaled_lon)]);
        }

        Ok(CaliMap {
            idMap: locations,
            dimensions: 2,
            name: "CaliMap".to_string(),
        })
    }
}

impl Searchable for CaliMap {
    // Specify the exact types used in CaliMap's HashMap
    type Key = u32;
    type Value = Vec<(u64, u64)>;

    fn get_dims(&self) -> i64 {
        self.dimensions
    }

    fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value> {
        &self.idMap
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
