use crate::dataloader::flatten_nd;
use crate::Record;
use crate::LAMA::utility::{compute_pair_weight, get_mbq, Distribution};
use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::{ConstraintProto, CpModelProto, TableConstraintProto};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Translator: One Formula from All Matching Pairs.
pub struct Translator {
    cp_model: CpModelBuilder,
    upper: i64,

    // Domain variables needed for on-the-fly math
    lowest_rec: Vec<i64>,
    largest_rec: Vec<i64>,
    dist_enum: Distribution, // Assumes this enum is in scope

    // OPTIMIZATION: HashMap -> Vec.
    // Assumes encrypted_id can be cast to usize safely (e.g., 0 to N).
    enc_id_to_intvar: Vec<Option<IntVar>>,

    // Kept as HashMap: We still need to map the opaque IntVar struct back to its protobuf index
    var_index_map: HashMap<IntVar, (i32, i64)>,
}

impl Translator {
    // UPDATED: Now takes domain bounds instead of the massive precomputed HashMap
    pub fn new(
        largest_val: i64,
        lowest_rec: Vec<i64>,
        largest_rec: Vec<i64>,
        dist_enum: Distribution,
        max_encrypted_id: usize, // Needed to pre-allocate the Vec
    ) -> Self {
        Self {
            cp_model: CpModelBuilder::default(),
            upper: largest_val,
            lowest_rec,
            largest_rec,
            dist_enum,
            // Pre-allocate the exact size needed
            enc_id_to_intvar: vec![None; max_encrypted_id + 1],
            var_index_map: HashMap::new(),
        }
    }

    pub fn set_all_vars(&mut self, encrypted_records: Vec<i64>) {
        let mut count = 0;
        let mut all_vars = Vec::new();

        let pb = ProgressBar::new(encrypted_records.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for encrypted_id in encrypted_records {
            let var = self
                .cp_model
                .new_int_var_with_name([(0, self.upper)], format!("rec_{encrypted_id}"));

            // FAST PATH: Array indexing instead of HashMap insertion
            self.enc_id_to_intvar[encrypted_id as usize] = Some(var);

            self.var_index_map.insert(var, (count, encrypted_id));
            all_vars.push(var);
            count += 1;
            pb.inc(1);
        }

        pb.finish();
        self.cp_model.add_all_different(all_vars);
    }

    fn find_matching_pairs(
        &self,
        observed_responses: &HashMap<(i64, u64), Vec<Vec<i64>>>,
    ) -> CpModelProto {
        // 1. EXTRACT TARGETS: Find the exact t-values and frequencies we actually care about
        // Group targets by 't' to avoid generating unused subset sizes
        let mut targets_by_t: HashMap<i64, HashSet<u64>> = HashMap::new();
        for &(t_val, freq) in observed_responses.keys() {
            targets_by_t.entry(t_val).or_default().insert(freq);
        }

        // 2. ON-THE-FLY PARALLEL CALCULATION
        println!("Computing required plaintexts on the fly...");
        let mut allowed_map: HashMap<(i64, u64), Vec<Vec<i64>>> = HashMap::new();

        // Generate the spatial grid once
        let mut vals: Vec<Record> = self
            .lowest_rec
            .iter()
            .zip(self.largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();
        vals.sort();

        for (t_val, target_freqs) in targets_by_t {
            let t_usize = t_val as usize;

            // Generate combinations, filter in parallel, and reduce into our allowed_map
            let t_map: HashMap<(i64, u64), Vec<Vec<i64>>> = vals
                .clone()
                .into_iter()
                .combinations(t_usize)
                .par_bridge() // Multithreaded execution starts here
                .filter_map(|val_tuple| {
                    let bounding_pair = get_mbq(&val_tuple);
                    let freq = compute_pair_weight(
                        &bounding_pair,
                        &self.dist_enum,
                        &self.lowest_rec,
                        &self.largest_rec,
                    );

                    // PRUNING: Only keep it if it matches a frequency we observed
                    if target_freqs.contains(&freq) {
                        let flattened: Vec<i64> = val_tuple
                            .iter()
                            .map(|rec| flatten_nd(rec, &self.largest_rec, &self.lowest_rec))
                            .collect();
                        Some(((t_val, freq), flattened))
                    } else {
                        None // Discard instantly (Saves massive amounts of memory)
                    }
                })
                .fold(HashMap::new, |mut local_map, (key, flat_tuple)| {
                    local_map.entry(key).or_default().push(flat_tuple);
                    local_map
                })
                .reduce(HashMap::new, |mut map1, map2| {
                    for (k, mut v) in map2 {
                        map1.entry(k).or_default().append(&mut v);
                    }
                    map1
                });

            allowed_map.extend(t_map);
        }

        // 3. APPLY TO CP-SAT MODEL (Your original logic exactly)
        let pb = ProgressBar::new(observed_responses.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut model = self.cp_model.proto().clone();

        for (key, t_tuples) in observed_responses {
            if let Some(allowed_plaintexts) = allowed_map.get(key) {
                for encrypted_tuple in t_tuples {
                    let mut vars = Vec::new();

                    for &encrypted_rec in encrypted_tuple {
                        // FAST PATH: Direct array lookup using the ID
                        if let Some(Some(var)) = self.enc_id_to_intvar.get(encrypted_rec as usize) {
                            vars.push(*var);
                        }
                    }

                    self.add_allowed_assignments(&mut model, vars, allowed_plaintexts.clone());
                }
            } else {
                panic!(
                    "Warning: Tuple {}, Freq {} not found (impossible frequency observed)",
                    key.0, key.1
                );
            }
            pb.inc(1);
        }
        pb.finish();
        model
    }

    pub fn translate(
        mut self,
        freq_record_match: &HashMap<(i64, u64), Vec<Vec<i64>>>,
        encrypted_records: Vec<i64>,
    ) -> (CpModelProto, HashMap<IntVar, (i32, i64)>) {
        self.set_all_vars(encrypted_records);
        let model = self.find_matching_pairs(freq_record_match);
        (model, self.var_index_map)
    }

    // This function remains 100% unchanged to preserve your workflow
    fn add_allowed_assignments(
        &self,
        model: &mut CpModelProto,
        vars: Vec<IntVar>,
        allowed_plaintexts: Vec<Vec<i64>>,
    ) {
        let mut table_proto = TableConstraintProto::default();

        for var in &vars {
            let var_index = self.var_index_map.get(var).expect("Unknown variable");
            table_proto.vars.push(var_index.0);
        }

        for tuple in allowed_plaintexts {
            for plaintext in tuple {
                table_proto.values.push(plaintext);
            }
        }

        let mut constraint_proto = ConstraintProto::default();
        constraint_proto.constraint = Some(Constraint::Table(table_proto));
        model.constraints.push(constraint_proto);
    }
}
