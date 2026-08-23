pub(crate) use crate::inst::{bank_matrix, decode, instruction};

use crate::inst::instruction::{BallInstruction, ExecContext};

#[path = "02_gemmini_config.rs"]
mod f02_gemmini_config;
#[path = "03_gemmini_flush.rs"]
mod f03_gemmini_flush;
#[path = "53_gemmini_preload.rs"]
mod f53_gemmini_preload;
#[path = "66_gemmini_compute_preloaded.rs"]
mod f66_gemmini_compute_preloaded;
#[path = "67_gemmini_compute_accumulated.rs"]
mod f67_gemmini_compute_accumulated;
#[path = "80_gemmini_loop_ws.rs"]
mod f80_gemmini_loop_ws;
#[path = "96_gemmini_loop_conv_ws.rs"]
mod f96_gemmini_loop_conv_ws;
mod gemmini_state;
mod loop_micro_ops;

const BALL_CLASS: &str = "examples.balls.gemmini.GemminiBall";

pub fn execute_known(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &mut ExecContext,
) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    let mnemonic = crate::config::ball_domain::mnemonic_for_funct(funct)?;
    let result = match mnemonic.as_str() {
        "GEMMINI_CONFIG" => f02_gemmini_config::GemminiConfig::exec(xs1, xs2, ctx),
        "GEMMINI_FLUSH" => f03_gemmini_flush::GemminiFlush::exec(xs1, xs2, ctx),
        "GEMMINI_PRELOAD" => f53_gemmini_preload::GemminiPreload::exec(xs1, xs2, ctx),
        "GEMMINI_COMPUTE_PRELOADED" => f66_gemmini_compute_preloaded::GemminiComputePreloaded::exec(xs1, xs2, ctx),
        "GEMMINI_COMPUTE_ACCUMULATED" => f67_gemmini_compute_accumulated::GemminiComputeAccumulated::exec(xs1, xs2, ctx),
        name if name.starts_with("GEMMINI_LOOP_WS_") || name == "GEMMINI_LOOP_WS" => f80_gemmini_loop_ws::execute(name, xs2, ctx),
        name if name.starts_with("GEMMINI_LOOP_CONV_WS_") || name == "GEMMINI_LOOP_CONV_WS" => f96_gemmini_loop_conv_ws::execute(name, xs2, ctx),
        _ => return None,
    };
    Some(result)
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    let mnemonic = crate::config::ball_domain::mnemonic_for_funct(funct)?;
    let latency = match mnemonic.as_str() {
        "GEMMINI_CONFIG" => f02_gemmini_config::GemminiConfig::latency(xs1, xs2),
        "GEMMINI_FLUSH" => f03_gemmini_flush::GemminiFlush::latency(xs1, xs2),
        "GEMMINI_PRELOAD" => f53_gemmini_preload::GemminiPreload::latency(xs1, xs2),
        "GEMMINI_COMPUTE_PRELOADED" => f66_gemmini_compute_preloaded::GemminiComputePreloaded::latency(xs1, xs2),
        "GEMMINI_COMPUTE_ACCUMULATED" => f67_gemmini_compute_accumulated::GemminiComputeAccumulated::latency(xs1, xs2),
        name if name.starts_with("GEMMINI_LOOP_WS_") || name == "GEMMINI_LOOP_WS" => f80_gemmini_loop_ws::latency(name),
        name if name.starts_with("GEMMINI_LOOP_CONV_WS_") || name == "GEMMINI_LOOP_CONV_WS" => f96_gemmini_loop_conv_ws::latency(name),
        _ => return None,
    };
    Some(latency)
}
