//===- gemmini_flush.rs - GEMMINI_FLUSH instruction ---------------------===//

use super::instruction::{BallModel, ExecContext};

pub struct GemminiFlush;

impl BallModel for GemminiFlush {
    fn exec(_xs1: u64, _xs2: u64, _ctx: &mut ExecContext) -> u64 {
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        super::latency::cycles(super::decode::rs1_iter(xs1), xs2, false, false, false)
    }
}
