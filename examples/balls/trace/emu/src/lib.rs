pub(crate) use crate::inst::instruction;

use crate::inst::instruction::{ExecContext, Instruction};

#[path = "04_bdb_counter.rs"]
mod f04_bdb_counter;

const BALL_CLASS: &str = "examples.balls.trace.TraceBall";

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
        <f04_bdb_counter::BdbCounter as Instruction>::FUNCT => Some(
            <f04_bdb_counter::BdbCounter as Instruction>::exec(xs1, xs2, ctx),
        ),
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match funct {
        <f04_bdb_counter::BdbCounter as Instruction>::FUNCT => Some(
            <f04_bdb_counter::BdbCounter as Instruction>::latency(xs1, xs2),
        ),
        _ => None,
    }
}
