pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::ExecContext;

mod smatmul;
mod mxfp8;
mod fp32;

const BALL_CLASS: &str = "examples.balls.smatmul.SMatMulBall";

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
        Some("SMATMUL_OS") => Some(smatmul::exec_smatmul(xs1, xs2, ctx)),
        Some("SMATMUL_MXFP8") => Some(mxfp8::execute(xs1, xs2, ctx)),
        Some("SMATMUL_F32") => Some(fp32::execute(xs1, xs2, ctx)),
        Some("SMATMUL_BIAS") => Some(smatmul::exec_bias(xs1, xs2, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("SMATMUL_OS") => Some(smatmul::latency(xs1, xs2)),
        Some("SMATMUL_MXFP8") => Some(smatmul::latency(xs1, xs2)),
        Some("SMATMUL_F32") => Some(decode::rs1_iter(xs1) as u64 + (xs2 & 0xfff) + ((xs2 >> 12) & 0xfff)),
        Some("SMATMUL_BIAS") => Some(smatmul::bias_latency(xs1, xs2)),
        _ => None,
    }
}
