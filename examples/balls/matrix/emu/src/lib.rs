pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::{ExecContext, Instruction};

#[path = "65_matrix.rs"]
mod f65_matrix;

const BALL_CLASS: &str = "examples.balls.matrix.MatrixBall";

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
        <f65_matrix::Matrix as Instruction>::FUNCT => {
            Some(<f65_matrix::Matrix as Instruction>::exec(xs1, xs2, ctx))
        }
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match funct {
        <f65_matrix::Matrix as Instruction>::FUNCT => {
            Some(<f65_matrix::Matrix as Instruction>::latency(xs1, xs2))
        }
        _ => None,
    }
}
