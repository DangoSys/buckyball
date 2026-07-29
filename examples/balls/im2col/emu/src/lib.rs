pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::{ExecContext, Instruction};

#[path = "48_im2col.rs"]
mod f48_im2col;

const BALL_CLASS: &str = "examples.balls.im2col.Im2colBall";

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
        <f48_im2col::Im2col as Instruction>::FUNCT => {
            Some(<f48_im2col::Im2col as Instruction>::exec(xs1, xs2, ctx))
        }
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match funct {
        <f48_im2col::Im2col as Instruction>::FUNCT => {
            Some(<f48_im2col::Im2col as Instruction>::latency(xs1, xs2))
        }
        _ => None,
    }
}
