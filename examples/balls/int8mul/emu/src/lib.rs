pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::{BallInstruction, ExecContext};

mod int8mul;

const BALL_CLASS: &str = "examples.balls.int8mul.Int8MulBall";

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
        Some("INT8MUL") => Some(int8mul::Int8Mul::exec(xs1, xs2, ctx)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("INT8MUL") => Some(int8mul::Int8Mul::latency(xs1, xs2)),
        Some(_) | None => None,
    }
}
