use crate::dataloader::datasets::Searchable;
use itertools::Itertools;

/// Selector: Choosing Record-Retrieval Events.
/// This component determines which record-retrieval set expressions are used.
/// Specifically, it generates the frequencies of dominating pairs, as it corresponds to some dist
pub struct Selector {
    dist: String,
    encrypted_db: dyn Searchable,
}

impl Selector {
    /// Generates T_cap combinations of size `t` for a given set of record IDs.
    pub fn generate_left_hand_expressions(records: &[u32], t: usize) -> Vec<Vec<u32>> {
        records.iter().cloned().combinations(t).collect()
    }

    // TODO: select based upon what values are 'plausible' for higher values of t.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_left_hand_expressions() {
        let records = vec![1, 2, 3, 4];
        let expressions = Selector::generate_left_hand_expressions(&records, 2);
        assert_eq!(expressions.len(), 6);
        assert_eq!(expressions[0], vec![1, 2]);
    }
}
