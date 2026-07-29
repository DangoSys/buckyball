pub(crate) use crate::inst::{bank_matrix, decode, instruction};

use crate::inst::instruction::{ExecContext, Instruction};

#[path = "64_mul_warp16.rs"]
mod f64_mul_warp16;

const BALL_CLASS: &str = "examples.balls.vector.VecBall";

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
    match funct {
        <f64_mul_warp16::MulWarp16 as Instruction>::FUNCT => Some(
            <f64_mul_warp16::MulWarp16 as Instruction>::exec(xs1, xs2, ctx),
        ),
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match funct {
        <f64_mul_warp16::MulWarp16 as Instruction>::FUNCT => Some(
            <f64_mul_warp16::MulWarp16 as Instruction>::latency(xs1, xs2),
        ),
        _ => None,
    }
}
