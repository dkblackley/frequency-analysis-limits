use crate::plotting::plot::ReconstructionData2dPoint;
use serde_json::json;
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn export_to_geojson(
    data: Vec<ReconstructionData2dPoint>,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut features = Vec::new();

    // 2. Map your data to standard GeoJSON features
    for (index, item) in data.iter().enumerate() {
        // GeoJSON strictly requires [longitude, latitude]
        // Ensure your f64 tuples are ordered correctly here!

        // Feature A: Ground Truth Point
        features.push(json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item.true_points.0, item.true_points.1]
            },
            "properties": {
                "pair_id": index,
                "point_type": "ground_truth"
            }
        }));

        // Feature B: Reconstructed Point
        features.push(json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item.reconstructed_points.0, item.reconstructed_points.1]
            },
            "properties": {
                "pair_id": index,
                "point_type": "reconstructed"
            }
        }));
    }

    // 3. Wrap it in a FeatureCollection
    let geojson = json!({
        "type": "FeatureCollection",
        "features": features
    });

    // 4. Write it out to the new file
    let mut output_file = File::create(output_path)?;
    let geojson_string = serde_json::to_string_pretty(&geojson)?;
    output_file.write_all(geojson_string.as_bytes())?;

    Ok(())
}
