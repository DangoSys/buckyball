pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::{BallInstruction, ExecContext};

#[path = "64_matadd.rs"]
mod f64_matadd;

const BALL_CLASS: &str = "examples.balls.matadd.MatAddBall";

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
        Some("MATADD") => Some(f64_matadd::MatAdd::exec(xs1, xs2, ctx)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MATADD") => Some(f64_matadd::MatAdd::latency(xs1, xs2)),
        Some(_) | None => None,
    }
}
