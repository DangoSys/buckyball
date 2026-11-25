use super::{latency, mxfp8};
use bebop_bemu::ball::blink::rs1_iter;
use bebop_bemu::ball::ExecContext;

const BALL_CLASS: &str = "examples.balls.mxmm.MxmmBall";

pub fn execute_known(ball_class: &str, funct: u32, xs1: u64, xs2: u64, ctx: &mut ExecContext) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("MXMM_MXFP8") => Some(mxfp8::execute(xs1, xs2, false, ctx)),
        Some("MXMM_MXFP8_WINDOW") => Some(mxfp8::execute(xs1, xs2, true, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    _ctx: &bebop_bemu::ExecContext,
) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("MXMM_MXFP8") => Some(cycles(xs1, xs2, None)),
        Some("MXMM_MXFP8_WINDOW") => Some(cycles(xs1, xs2, Some(((xs2 >> 32) & 0xffff, xs2 >> 48)))),
        _ => None,
    }
}

// rs2: M [11:0], N [23:12], first [24], last [25]; a window adds fullK [47:32]
// and startK [63:48].
fn cycles(xs1: u64, xs2: u64, window: Option<(u64, u64)>) -> u64 {
    latency::cycles(xs2 & 0xfff, (xs2 >> 12) & 0xfff, rs1_iter(xs1), xs2 & (1 << 24) != 0, xs2 & (1 << 25) != 0, window)
}
