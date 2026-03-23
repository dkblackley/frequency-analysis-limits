use crate::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::BufWriter;

// From lilika's paper

// Database 1: Users
#[derive(Debug)]
pub struct CaliMap50 {
    //Map of 'node_id' to the two lat and longs (multiplied by 100 and cast to u64) - inner vec should
    // always be of size 2
    plaintext: HashMap<u32, Vec<Vec<Value>>>,

    grid: Vec<Vec<Value>>,
    dimensions: Value, // should always be two
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Location {
    id: u32,
    longitude: f64,
    latitude: f64,
}

// impl CaliMap50 {
//     pub fn new(file_path: &str) -> Result<Self, DataLoadingError> {
//         let file = File::open(file_path)?;
//
//         // Configure the CSV reader
//         let mut rdr = ReaderBuilder::new()
//             .has_headers(false) // Critical: tells the parser not to skip the first row
//             .delimiter(b'\t') // Set the delimiter to a tab (use b' ' if it's actually space-separated)
//             .from_reader(file);
//
//         let mut locations = HashMap::new();
//         // Iterate over the rows and deserialize them into our struct
//         for result in rdr.deserialize() {
//             // The `?` operator unpackages the Result, returning an error if parsing fails
//             let record: Location = result.map_err(|csv_err| DataLoadingError::Parsing {
//                 file_path: file_path.to_string(),
//                 error: csv_err.to_string(),
//             })?;
//
//             // You now have a strongly typed struct
//             trace!(
//                 "ID: {}, Lat: {}, Lon: {}",
//                 record.id, record.latitude, record.longitude
//             );
//             let scaled_lat = (record.latitude * 100.0) as u64;
//             let scaled_lon = (record.longitude * 100.0) as u64;
//
//             locations.insert(record.id, vec![(scaled_lat, scaled_lon)]);
//         }
//
//         Ok(CaliMap50 {
//             plaintext: locations,
//             dimensions: 2,
//             name: "CaliMap".to_string(),
//         })
//     }

// pub fn new(file_path: &str) -> Result<Self, DataLoadingError> {
//     let file = File::open(file_path)?;
//
//     // Configure the CSV reader
//     let mut rdr = ReaderBuilder::new()
//         .has_headers(false) // Critical: tells the parser not to skip the first row
//         .delimiter(b'\t') // Set the delimiter to a tab (use b' ' if it's actually space-separated)
//         .from_reader(file);
//
//     let mut locations = HashMap::new();
//     // Iterate over the rows and deserialize them into our struct
//     for result in rdr.deserialize() {
//         // The `?` operator unpackages the Result, returning an error if parsing fails
//         let record: Location = result.map_err(|csv_err| DataLoadingError::Parsing {
//             file_path: file_path.to_string(),
//             error: csv_err.to_string(),
//         })?;
//
//         // You now have a strongly typed struct
//         trace!(
//             "ID: {}, Lat: {}, Lon: {}",
//             record.id, record.latitude, record.longitude
//         );
//         let scaled_lat = (record.latitude * 100.0) as u64;
//         let scaled_lon = (record.longitude * 100.0) as u64;
//
//         locations.insert(record.id, vec![(scaled_lat, scaled_lon)]);
//     }
//
//     Ok(CaliMap50 {
//         plaintext: locations,
//         dimensions: 2,
//         name: "CaliMap".to_string(),
//     })
// }
//}

// Note: A `Vec` requires heap allocation and cannot be a `const` in standard Rust.
// Assuming `CALI_ALL` is either flattened or handled as a slice `&[(i32, i32)]`.
pub fn scale_points(points: &[(i32, i32)], n0: i32, n1: i32) -> Vec<(i32, i32)> {
    let mut max_n0 = 0;
    let mut max_n1 = 0;

    for &(i, j) in points {
        max_n0 = max_n0.max(i);
        max_n1 = max_n1.max(j);
    }

    // Prevent division by zero
    if max_n0 == 0 {
        max_n0 = 1;
    }
    if max_n1 == 0 {
        max_n1 = 1;
    }

    points
        .iter()
        .map(|&(i, j)| {
            let new_i = 1.max(i * n0 / max_n0);
            let new_j = 1.max(j * n1 / max_n1);
            (new_i, new_j)
        })
        .collect()
}

pub fn map_to_locations(points: &[(i32, i32)]) -> Vec<Location> {
    points
        .iter()
        .enumerate()
        .map(|(id, &(longitude, latitude))| Location {
            id: id as u32,
            longitude: longitude as f64,
            latitude: latitude as f64,
        })
        .collect()
}

pub fn save_locations_to_file(
    locations: &[Location],
    filepath: &str,
    dictpath: &str,
) -> Result<(), Box<dyn Error>> {
    match fs::create_dir_all(dictpath) {
        Ok(_) => println!("Directory created successfully (or it already exists)!"),
        Err(e) => println!("Failed to create directory: {}", e),
    }
    let file = File::create(filepath)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, locations)?;
    Ok(())
}

// impl Searchable for CaliMap50 {
//     // Specify the exact types used in CaliMap's HashMap
//     type Value = u32;
//     type Record = Vec<(u64, u64)>;
//
//     fn get_dims(&self) -> i64 {
//         self.dimensions
//     }
//
//     // fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value> {
//     //     &self.idMap
//     // }
//
//     fn get_name(&self) -> &str {
//         &self.name
//     }
//
//     fn do_search(&self, query: &(Self::Record, Self::Record)) -> Vec<Self::Record> {
//         todo!()
//     }
// }
