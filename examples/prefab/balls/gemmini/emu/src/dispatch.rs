use bebop_bemu::ball::{BallModel, ExecContext};

use super::decode::rs1_iter;
use super::gemmini_state::{GemminiCfg, GemminiState};
use super::{
    gemmini_config, gemmini_flush, gemmini_preload, gemmini_compute_preloaded,
    gemmini_compute_accumulated, gemmini_loop_ws, gemmini_loop_conv_ws, latency,
};

const BALL_CLASS: &str = "examples.balls.gemmini.GemminiBall";

pub fn execute_known(ball_class: &str, funct: u32, xs1: u64, xs2: u64, ctx: &mut ExecContext) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    let mnemonic = bebop_bemu::ball::blink::mnemonic_for_funct(funct)?;
    let result = match mnemonic.as_str() {
        "GEMMINI_CONFIG" => gemmini_config::GemminiConfig::exec(xs1, xs2, ctx),
        "GEMMINI_FLUSH" => gemmini_flush::GemminiFlush::exec(xs1, xs2, ctx),
        "GEMMINI_PRELOAD" => gemmini_preload::GemminiPreload::exec(xs1, xs2, ctx),
        "GEMMINI_COMPUTE_PRELOADED" => gemmini_compute_preloaded::GemminiComputePreloaded::exec(xs1, xs2, ctx),
        "GEMMINI_COMPUTE_ACCUMULATED" => {
            gemmini_compute_accumulated::GemminiComputeAccumulated::exec(xs1, xs2, ctx)
        }
        name if name.starts_with("GEMMINI_LOOP_WS_") || name == "GEMMINI_LOOP_WS" => {
            gemmini_loop_ws::execute(name, xs2, ctx)
        }
        name if name.starts_with("GEMMINI_LOOP_CONV_WS_") || name == "GEMMINI_LOOP_CONV_WS" => {
            gemmini_loop_conv_ws::execute(name, xs2, ctx)
        }
        _ => return None,
    };
    Some(result)
}

pub fn cycles_after_issue(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &bebop_bemu::ExecContext,
) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    let mnemonic = bebop_bemu::ball::blink::mnemonic_for_funct(funct)?;
    let latency = match mnemonic.as_str() {
        "GEMMINI_CONFIG" => gemmini_config::GemminiConfig::latency(xs1, xs2),
        "GEMMINI_FLUSH" => gemmini_flush::GemminiFlush::latency(xs1, xs2),
        // Their cycles follow the configuration the last GEMMINI_CONFIG set.
        "GEMMINI_PRELOAD" | "GEMMINI_COMPUTE_PRELOADED" | "GEMMINI_COMPUTE_ACCUMULATED" => {
            let cfg = configuration(ctx);
            latency::cycles(rs1_iter(xs1), xs2, cfg.dataflow == 1, cfg.a_transpose, cfg.b_transpose)
        }
        // A loop: its expansion, then each Ball instruction it expands into.
        "GEMMINI_LOOP_WS" => {
            let cfg = configuration(ctx);
            let bounds = ctx.metadata_word(0);
            let (i, j, k) = (bounds >> 32 & 0xffff, bounds >> 16 & 0xffff, bounds & 0xffff);
            latency::loop_ws_cycles(i, j, k)
                + latency::loop_ws_ball_cycles(i, j, k, cfg.dataflow == 1, cfg.a_transpose, cfg.b_transpose)
        }
        "GEMMINI_LOOP_CONV_WS" => {
            let cfg = configuration(ctx);
            let (w0, w1, w2) = (ctx.metadata_word(64), ctx.metadata_word(72), ctx.metadata_word(80));
            let (batch, out_dim, out_channels) = (w0 & 0xffff, w1 >> 16 & 0xffff, w1 & 0xffff);
            let (in_channels, kernel_dim) = (w0 >> 32 & 0xffff, w2 & 0xff);
            latency::loop_conv_ws_cycles(batch, out_dim, out_channels, in_channels, kernel_dim)
                + latency::loop_conv_ws_ball_cycles(
                    batch,
                    out_dim,
                    out_channels,
                    in_channels,
                    kernel_dim,
                    (xs2 >> 30) & 1 == 1,
                    cfg.dataflow == 1,
                    cfg.a_transpose,
                    cfg.b_transpose,
                )
        }
        name if name.starts_with("GEMMINI_LOOP_WS_CONFIG_") || name.starts_with("GEMMINI_LOOP_CONV_WS_CONFIG_") => {
            latency::loop_config_cycles()
        }
        _ => return None,
    };
    Some(latency)
}

/// The configuration of this Ball: its reset value until a GEMMINI_CONFIG runs.
fn configuration(ctx: &bebop_bemu::ExecContext) -> GemminiCfg {
    let key = (
        ctx.ball_id.expect("Gemmini cycles need the issued Ball"),
        std::any::TypeId::of::<GemminiState>(),
    );
    match ctx.ball_states.get(&key) {
        Some(state) => state
            .downcast_ref::<std::sync::Arc<std::sync::Mutex<GemminiState>>>()
            .expect("Gemmini state type")
            .lock()
            .unwrap()
            .cfg
            .clone(),
        None => GemminiCfg::default(),
    }
}
