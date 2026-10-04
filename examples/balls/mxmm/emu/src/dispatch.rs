use super::{fp32, mxfp8};
use crate::inst::instruction::ExecContext;

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
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXMM_MXFP8") => Some(mxfp8::execute(xs1, xs2, false, ctx)),
        Some("MXMM_MXFP8_WINDOW") => Some(mxfp8::execute(xs1, xs2, true, ctx)),
        Some("MXMM_FMA32") => Some(fp32::execute(xs1, xs2, true, ctx)),
        Some("MXMM_F32") => Some(fp32::execute(xs1, xs2, false, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, _xs1: u64, _xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXMM_MXFP8" | "MXMM_MXFP8_WINDOW" | "MXMM_FMA32" | "MXMM_F32") => Some(1),
        _ => None,
    }
}
