pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::{BallInstruction, ExecContext};

mod int8add;

const BALL_CLASS: &str = "examples.balls.int8add.Int8AddBall";

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
        Some("INT8ADD") => Some(int8add::Int8Add::exec(xs1, xs2, ctx, false)),
        Some("INT8ADD_RELU") => Some(int8add::Int8Add::exec(xs1, xs2, ctx, true)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("INT8ADD") | Some("INT8ADD_RELU") => Some(int8add::Int8Add::latency(xs1, xs2)),
        Some(_) | None => None,
    }
}
