pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::{ExecContext, Instruction};

#[path = "55_mxfp2int.rs"]
mod f55_mxfp2int;

const BALL_CLASS: &str = "examples.balls.mxfp2int.Mxfp2IntBall";

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
        <f55_mxfp2int::Mxfp2Int as Instruction>::FUNCT => {
            Some(<f55_mxfp2int::Mxfp2Int as Instruction>::exec(xs1, xs2, ctx))
        }
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match funct {
        <f55_mxfp2int::Mxfp2Int as Instruction>::FUNCT => {
            Some(<f55_mxfp2int::Mxfp2Int as Instruction>::latency(xs1, xs2))
        }
        _ => None,
    }
}
