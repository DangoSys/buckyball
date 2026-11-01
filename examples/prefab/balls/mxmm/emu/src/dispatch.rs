use super::{accumulator::Format, fp32, mxfp8, timing};
use bebop_bemu::ball::ExecContext;

const BALL_CLASS: &str = "examples.balls.mxmm.MxmmBall";

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
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXMM_MXFP8") => Some(mxfp8::execute(xs1, xs2, false, ctx)),
        Some("MXMM_MXFP8_WINDOW") => Some(mxfp8::execute(xs1, xs2, true, ctx)),
        Some("MXMM_FMA32") => Some(fp32::execute(xs1, xs2, true, ctx)),
        Some("MXMM_F32") => Some(fp32::execute(xs1, xs2, false, ctx)),
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
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXMM_MXFP8") => Some(timing::latency(xs1, xs2, Format::Mxfp8)),
        Some("MXMM_MXFP8_WINDOW") => Some(timing::latency(xs1, xs2, Format::Mxfp8Window)),
        Some("MXMM_FMA32") => Some(timing::latency(xs1, xs2, Format::Fma32)),
        Some("MXMM_F32") => Some(timing::latency(xs1, xs2, Format::F32)),
        _ => None,
    }
}
