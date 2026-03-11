use itertools::Itertools;

/// Selector: Choosing Record-Retrieval Events[cite: 181].
/// This component determines which record-retrieval set expressions are used.
/// Specifically, it generates the left-hand expressions (EX_L) as a collection[cite: 184].
pub struct Selector;

impl Selector {
    /// Generates T_cap combinations of size `t` for a given set of record IDs.
    pub fn generate_left_hand_expressions(records: &[u32], t: usize) -> Vec<Vec<u32>> {
        records.iter().cloned().combinations(t).collect()
    }
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
