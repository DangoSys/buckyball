//===- 03_gemmini_flush.rs - GEMMINI_FLUSH instruction ---------------------===//

use super::instruction::{BallInstruction, ExecContext};

pub struct GemminiFlush;

impl BallInstruction for GemminiFlush {
    fn exec(_xs1: u64, _xs2: u64, _ctx: &mut ExecContext) -> u64 {
        0
    }

    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        1
    }
}
