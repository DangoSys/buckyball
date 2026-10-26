//===- 02_gemmini_config.rs - GEMMINI_CONFIG instruction -------------------===//

use super::gemmini_state::gemini;
use super::instruction::{BallModel, ExecContext};

pub struct GemminiConfig;

impl BallModel for GemminiConfig {
    fn exec(_xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let state = gemini(ctx);
        let mut g = state.lock().unwrap();
        g.cfg.dataflow = ((xs2 >> 4) & 1) as u8;
        g.cfg.a_transpose = ((xs2 >> 7) & 1) != 0;
        g.cfg.b_transpose = ((xs2 >> 8) & 1) != 0;
        g.cfg.in_shift = ((xs2 >> 9) & 0xFFFFFFFF) as u32;
        0
    }

    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        1
    }
}
