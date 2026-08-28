pub(crate) use crate::inst::decode;
pub(crate) use crate::inst::instruction;

use crate::inst::instruction::ExecContext;

mod fp2int;
mod model;

#[cfg(test)]
mod tests;

const BALL_CLASS: &str = "examples.balls.fp2int.Fp2IntBall";

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
        Some("FP2INT") => Some(fp2int::exec_fp2int(xs1, xs2, ctx)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("FP2INT") => Some(fp2int::latency(xs1)),
        Some(_) | None => None,
    }
}
