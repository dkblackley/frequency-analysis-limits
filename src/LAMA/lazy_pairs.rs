//! Perfect setting (--query-percent 1.0), t = 2, WITHOUT materialising the pair tables.
//!
//! `Translator::process_cpsat_global` builds, for every pair of records (i, j), the table of
//! plaintext pairs (a, b) with a in C_i, b in C_j, a != b and `validate_candidate([enc_i, enc_j],
//! [a, b])` true, and hands all of them to Pumpkin as `table` constraints. Pumpkin encodes every
//! row as a fresh literal plus clauses, and `CpModel::solve` builds that model once per root value
//! on every thread. With noisy distributions (eps ~ 1e-3 at 50x50) that is millions of rows and the
//! jobs OOM at 64 GB within seconds (e.g. Shanghai uniform TV 0.01: 6.5M rows).
//!
//! Here the same `validate_candidate` answers are stored as ONE BIT per (a, b) candidate pair, shared
//! read-only by every partition through an `Arc`, and the tables are replaced by a single Pumpkin
//! propagator (`LazyPairTable`): when a record is fixed to a value, every value of every other record
//! that forms a disallowed pair with it is removed. This is forward checking on exactly the same
//! constraint: the search is still a complete Pumpkin enumeration, so the SET of solutions is the
//! same as with the tables (only the order Pumpkin finds them in can differ). Memory: O(sum |C_i| x
//! sum |C_j| / 2) bits instead of ~70 bytes per valid row plus Pumpkin's literals and clauses per
//! partition. CPU: the same validate_candidate calls as building the tables; propagation is
//! O(sum_j |C_j|) per fixed record.
//!
//! Semantics copied from process_cpsat_global, deliberately: only records that have t = 1 candidates
//! take part, and a pair with NO valid row gets no constraint at all (process_cpsat_global skips empty
//! tables). Opt-in: --lazy-pairs (default is the table path). Tests: `perfect_lazy_pairs_match_tables` in
//! tests/integration.rs.

use log::info;
use pumpkin_solver::core::proof::{ConstraintTag, InferenceCode};
use pumpkin_solver::core::propagation::{
    DomainEvents, EnqueueDecision, LocalId, NotificationContext, OpaqueDomainEvent,
    PropagationContext, Propagator, PropagatorConstructor, PropagatorConstructorContext,
    ReadDomains,
};
use pumpkin_solver::core::state::PropagationStatusCP;
use pumpkin_solver::core::variables::DomainId;
use pumpkin_solver::core::{conjunction, declare_inference_label, predicate};
use rayon::prelude::*;
use std::sync::Arc;

/// Which (a, b) value pairs each pair of records allows. Records are indexed by their solver
/// variable index. Bits of the pair (i, j), i < j, start at `offsets[tri(i, j)]`; bit
/// `ia * |C_j| + ib` is set iff (cands[i][ia], cands[j][ib]) is a row of that pair's table.
pub struct PairBits {
    n: usize,
    /// Sorted t = 1 candidate values per record (empty: record has no candidates, takes no part)
    cands: Vec<Vec<i64>>,
    /// u64::MAX: no constraint between the two records (no candidates, or an empty table)
    offsets: Vec<u64>,
    bits: Vec<u64>,
}

/// What building the bits found, for the translator's bookkeeping (same numbers the tables gave).
pub struct PairBuild {
    pub bits: PairBits,
    /// Number of table rows (set bits) = what process_cpsat_global counts as constraints at t = 2
    pub rows: usize,
    /// [enc_lo, enc_hi] of every constrained pair (non-empty table), enc_lo < enc_hi
    pub enc_tuples: Vec<Vec<i64>>,
    /// Records whose true assignment appears in one of their pair tables, with its error (debug only)
    pub found: Vec<(i64, f64)>,
}

impl std::fmt::Debug for PairBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PairBits {{ n: {}, bits: {} words }}", self.n, self.bits.len())
    }
}

impl PairBits {
    fn tri(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        i * self.n - i * (i + 1) / 2 + (j - i - 1)
    }

    /// Offset of the pair's bits, None if the two records are unconstrained.
    fn offset(&self, i: usize, j: usize) -> Option<u64> {
        let o = if i < j { self.offsets[self.tri(i, j)] } else { self.offsets[self.tri(j, i)] };
        (o != u64::MAX).then_some(o)
    }

    /// Is (value index ia of record i, value index ib of record j) allowed? (pair is constrained)
    fn allowed(&self, off: u64, i: usize, ia: usize, j: usize, ib: usize) -> bool {
        let bit = if i < j {
            off + (ia * self.cands[j].len() + ib) as u64
        } else {
            off + (ib * self.cands[i].len() + ia) as u64
        };
        self.bits[(bit / 64) as usize] >> (bit % 64) & 1 == 1
    }

    /// One bit per candidate pair, from exactly the calls process_cpsat_global makes:
    /// validate([enc_lo, enc_hi], [a_lo, a_hi]) with enc_lo < enc_hi, skipping a == b.
    pub fn build<V>(enc_of_var: Vec<i64>, cands: Vec<Vec<i64>>, validate: &V) -> PairBuild
    where
        V: Fn(&[i64], &[i64]) -> (bool, f64) + Sync + Send,
    {
        let n = enc_of_var.len();
        // Per row i: its pairs (i, j > i), bits word-aligned per row so rows build in parallel
        struct Row {
            words: Vec<u64>,
            local: Vec<u64>, // per j > i: bit offset within this row, or u64::MAX
            rows: usize,
            found: Vec<(i64, f64)>,
        }
        let built: Vec<Row> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = Row { words: Vec::new(), local: Vec::with_capacity(n - i - 1), rows: 0, found: Vec::new() };
                let mut next = 0u64;
                for j in i + 1..n {
                    if cands[i].is_empty() || cands[j].is_empty() {
                        row.local.push(u64::MAX);
                        continue;
                    }
                    let start = next;
                    let len = (cands[i].len() * cands[j].len()) as u64;
                    next += len;
                    row.words.resize(next.div_ceil(64) as usize, 0);
                    let (ei, ej) = (enc_of_var[i], enc_of_var[j]);
                    let swap = ej < ei; // validate sees the tuple in encrypted-id order
                    let enc = if swap { [ej, ei] } else { [ei, ej] };
                    let mut any = false;
                    for (ia, &a) in cands[i].iter().enumerate() {
                        for (ib, &b) in cands[j].iter().enumerate() {
                            if a == b {
                                continue;
                            }
                            let pt = if swap { [b, a] } else { [a, b] };
                            let (valid, err) = validate(&enc, &pt);
                            if valid {
                                let bit = start + (ia * cands[j].len() + ib) as u64;
                                row.words[(bit / 64) as usize] |= 1 << (bit % 64);
                                row.rows += 1;
                                any = true;
                                if (a == ei && b == ej) || (a == ej && b == ei) {
                                    row.found.push((ei, err));
                                    row.found.push((ej, err));
                                }
                            }
                        }
                    }
                    if any {
                        row.local.push(start);
                    } else {
                        // empty table: process_cpsat_global adds no constraint for this pair
                        row.local.push(u64::MAX);
                        next = start;
                        row.words.truncate(next.div_ceil(64) as usize);
                    }
                }
                row
            })
            .collect();

        let mut offsets = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        let mut bits = Vec::new();
        let mut rows = 0;
        let mut enc_tuples = Vec::new();
        let mut found = Vec::new();
        for (i, row) in built.into_iter().enumerate() {
            let base = bits.len() as u64 * 64;
            for (k, &o) in row.local.iter().enumerate() {
                if o == u64::MAX {
                    offsets.push(u64::MAX);
                } else {
                    offsets.push(base + o);
                    let (ei, ej) = (enc_of_var[i], enc_of_var[i + 1 + k]);
                    enc_tuples.push(vec![ei.min(ej), ei.max(ej)]);
                }
            }
            bits.extend_from_slice(&row.words);
            rows += row.rows;
            found.extend(row.found);
        }
        info!(
            "Lazy pair tables: {} constrained pairs, {} allowed rows, {:.1} MB of bits (no tables built)",
            enc_tuples.len(),
            rows,
            bits.len() as f64 * 8.0 / 1e6
        );
        PairBuild { bits: PairBits { n, cands, offsets, bits }, rows, enc_tuples, found }
    }
}

declare_inference_label!(LazyPairTableLabel);

/// Constructor: posted by `CpModel::solve_partition` when the model carries `lazy_pairs`.
#[derive(Clone)]
pub struct LazyPairTable {
    pub vars: Vec<DomainId>,
    pub pairs: Arc<PairBits>,
    pub constraint_tag: ConstraintTag,
}

impl PropagatorConstructor for LazyPairTable {
    type PropagatorImpl = LazyPairTablePropagator;

    fn create(self, mut context: PropagatorConstructorContext) -> Self::PropagatorImpl {
        for (i, var) in self.vars.iter().enumerate() {
            context.register(*var, DomainEvents::ASSIGN, LocalId::from(i as u32));
        }
        LazyPairTablePropagator {
            pending: (0..self.vars.len()).collect(), // first call: whatever is fixed at the root
            vars: self.vars,
            pairs: self.pairs,
            inference_code: InferenceCode::new(self.constraint_tag, LazyPairTableLabel),
        }
    }
}

#[derive(Clone)]
pub struct LazyPairTablePropagator {
    vars: Vec<DomainId>,
    pairs: Arc<PairBits>,
    /// Records assigned since the last propagation (re-checking one is harmless, skipping one is not)
    pending: Vec<usize>,
    inference_code: InferenceCode,
}

impl LazyPairTablePropagator {
    /// Record i is fixed to a: remove from every constrained partner j each value b with (a, b) not a
    /// row of the pair's table. Partners that become fixed are queued in `work`.
    fn process(&self, i: usize, context: &mut PropagationContext, work: &mut Vec<usize>) -> PropagationStatusCP {
        let xi = self.vars[i];
        let Some(a) = context.fixed_value(&xi) else { return Ok(()) };
        let p = &*self.pairs;
        let ia = p.cands[i].binary_search(&(a as i64)).ok();
        for j in 0..p.n {
            if j == i {
                continue;
            }
            let Some(off) = p.offset(i, j) else { continue };
            let xj = self.vars[j];
            for (ib, &b) in p.cands[j].iter().enumerate() {
                let b = b as i32;
                if !context.contains(&xj, b) {
                    continue;
                }
                if ia.is_some_and(|ia| p.allowed(off, i, ia, j, ib)) {
                    continue;
                }
                context.post(predicate!(xj != b), (conjunction!([xi == a]), &self.inference_code))?;
                if context.is_fixed(&xj) {
                    work.push(j);
                }
            }
            // values of j outside C_j are removed by j's t = 1 table
        }
        Ok(())
    }
}

impl Propagator for LazyPairTablePropagator {
    fn name(&self) -> &str {
        "LazyPairTable"
    }

    fn notify(&mut self, _context: NotificationContext, local_id: LocalId, _event: OpaqueDomainEvent) -> EnqueueDecision {
        self.pending.push(local_id.unpack() as usize);
        EnqueueDecision::Enqueue
    }

    fn propagate(&mut self, mut context: PropagationContext) -> PropagationStatusCP {
        let mut work = std::mem::take(&mut self.pending);
        while let Some(i) = work.pop() {
            self.process(i, &mut context, &mut work)?;
        }
        Ok(())
    }

    fn propagate_from_scratch(&self, mut context: PropagationContext) -> PropagationStatusCP {
        let mut work: Vec<usize> = (0..self.vars.len()).collect();
        while let Some(i) = work.pop() {
            self.process(i, &mut context, &mut work)?;
        }
        Ok(())
    }
}
