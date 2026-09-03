pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::ExecContext;

mod int2fp;

const BALL_CLASS: &str = "examples.balls.int2fp.Int2FpBall";

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
        Some("INT32_TO_FP32") => Some(int2fp::execute(xs1, xs2, ctx)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("INT32_TO_FP32") => Some(int2fp::latency(xs1, xs2)),
        Some(_) | None => None,
    }
}
