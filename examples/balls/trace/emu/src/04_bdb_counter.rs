//===- 04_bdb_counter.rs - BDB_COUNTER instruction -------------------------===//

use super::instruction::{BallInstruction, ExecContext};

pub struct BdbCounter;

impl BallInstruction for BdbCounter {
    fn exec(_xs1: u64, _xs2: u64, _ctx: &mut ExecContext) -> u64 {
        0
    }

    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        1
    }
}
