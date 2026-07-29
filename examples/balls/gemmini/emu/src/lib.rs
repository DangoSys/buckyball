pub(crate) use crate::inst::{bank_matrix, decode, instruction};

use crate::inst::instruction::{ExecContext, Instruction};

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
    match funct {
        <f02_gemmini_config::GemminiConfig as Instruction>::FUNCT => Some(
            <f02_gemmini_config::GemminiConfig as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f03_gemmini_flush::GemminiFlush as Instruction>::FUNCT => Some(
            <f03_gemmini_flush::GemminiFlush as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f53_gemmini_preload::GemminiPreload as Instruction>::FUNCT => Some(
            <f53_gemmini_preload::GemminiPreload as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f66_gemmini_compute_preloaded::GemminiComputePreloaded as Instruction>::FUNCT => Some(
            <f66_gemmini_compute_preloaded::GemminiComputePreloaded as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f67_gemmini_compute_accumulated::GemminiComputeAccumulated as Instruction>::FUNCT => Some(
            <f67_gemmini_compute_accumulated::GemminiComputeAccumulated as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigBounds as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigBounds as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrA as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrA as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrB as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrB as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrD as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrD as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrC as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrC as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesAB as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesAB as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesDC as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesDC as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWs as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWs as Instruction>::exec(xs1, xs2, ctx),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig1 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig1 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig2 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig2 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig3 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig3 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig4 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig4 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig5 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig5 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig6 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig6 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig7 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig7 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig8 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig8 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig9 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig9 as Instruction>::exec(
                xs1, xs2, ctx,
            ),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWs as Instruction>::FUNCT => {
            Some(<f96_gemmini_loop_conv_ws::GemminiLoopConvWs as Instruction>::exec(xs1, xs2, ctx))
        }
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match funct {
        <f02_gemmini_config::GemminiConfig as Instruction>::FUNCT => Some(
            <f02_gemmini_config::GemminiConfig as Instruction>::latency(xs1, xs2),
        ),
        <f03_gemmini_flush::GemminiFlush as Instruction>::FUNCT => Some(
            <f03_gemmini_flush::GemminiFlush as Instruction>::latency(xs1, xs2),
        ),
        <f53_gemmini_preload::GemminiPreload as Instruction>::FUNCT => {
            Some(<f53_gemmini_preload::GemminiPreload as Instruction>::latency(xs1, xs2))
        }
        <f66_gemmini_compute_preloaded::GemminiComputePreloaded as Instruction>::FUNCT => Some(
            <f66_gemmini_compute_preloaded::GemminiComputePreloaded as Instruction>::latency(
                xs1, xs2,
            ),
        ),
        <f67_gemmini_compute_accumulated::GemminiComputeAccumulated as Instruction>::FUNCT => Some(
            <f67_gemmini_compute_accumulated::GemminiComputeAccumulated as Instruction>::latency(
                xs1, xs2,
            ),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigBounds as Instruction>::FUNCT => {
            Some(<f80_gemmini_loop_ws::GemminiLoopWsConfigBounds as Instruction>::latency(xs1, xs2))
        }
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrA as Instruction>::FUNCT => {
            Some(<f80_gemmini_loop_ws::GemminiLoopWsConfigAddrA as Instruction>::latency(xs1, xs2))
        }
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrB as Instruction>::FUNCT => {
            Some(<f80_gemmini_loop_ws::GemminiLoopWsConfigAddrB as Instruction>::latency(xs1, xs2))
        }
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrD as Instruction>::FUNCT => {
            Some(<f80_gemmini_loop_ws::GemminiLoopWsConfigAddrD as Instruction>::latency(xs1, xs2))
        }
        <f80_gemmini_loop_ws::GemminiLoopWsConfigAddrC as Instruction>::FUNCT => {
            Some(<f80_gemmini_loop_ws::GemminiLoopWsConfigAddrC as Instruction>::latency(xs1, xs2))
        }
        <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesAB as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesAB as Instruction>::latency(xs1, xs2),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesDC as Instruction>::FUNCT => Some(
            <f80_gemmini_loop_ws::GemminiLoopWsConfigStridesDC as Instruction>::latency(xs1, xs2),
        ),
        <f80_gemmini_loop_ws::GemminiLoopWs as Instruction>::FUNCT => {
            Some(<f80_gemmini_loop_ws::GemminiLoopWs as Instruction>::latency(xs1, xs2))
        }
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig1 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig1 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig2 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig2 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig3 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig3 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig4 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig4 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig5 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig5 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig6 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig6 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig7 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig7 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig8 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig8 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig9 as Instruction>::FUNCT => Some(
            <f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig9 as Instruction>::latency(xs1, xs2),
        ),
        <f96_gemmini_loop_conv_ws::GemminiLoopConvWs as Instruction>::FUNCT => {
            Some(<f96_gemmini_loop_conv_ws::GemminiLoopConvWs as Instruction>::latency(xs1, xs2))
        }
        _ => None,
    }
}
