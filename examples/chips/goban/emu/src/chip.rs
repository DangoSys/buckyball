pub(crate) mod bank {
    pub(crate) use crate::bank::*;
}

use crate::inst::instruction::ExecContext;

#[path = "../../../../balls/fp2int/emu/src/lib.rs"]
mod fp2int;
#[path = "../../../../balls/im2col/emu/src/lib.rs"]
mod im2col;
#[path = "../../../../balls/int2fp/emu/src/lib.rs"]
mod int2fp;
#[path = "../../../../balls/transpose/emu/src/lib.rs"]
mod transpose;
#[path = "../../../../balls/vector/emu/src/lib.rs"]
mod vector;

pub fn execute_known(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &mut ExecContext,
) -> u64 {
    fp2int::execute_known(ball_class, funct, xs1, xs2, ctx)
        .or_else(|| im2col::execute_known(ball_class, funct, xs1, xs2, ctx))
        .or_else(|| int2fp::execute_known(ball_class, funct, xs1, xs2, ctx))
        .or_else(|| transpose::execute_known(ball_class, funct, xs1, xs2, ctx))
        .or_else(|| vector::execute_known(ball_class, funct, xs1, xs2, ctx))
        .unwrap_or_else(|| {
            panic!("no goban BEMU ball implementation for ballClass={ball_class} funct7={funct}")
        })
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> u64 {
    fp2int::cycles_after_issue(ball_class, funct, xs1, xs2)
        .or_else(|| im2col::cycles_after_issue(ball_class, funct, xs1, xs2))
        .or_else(|| int2fp::cycles_after_issue(ball_class, funct, xs1, xs2))
        .or_else(|| transpose::cycles_after_issue(ball_class, funct, xs1, xs2))
        .or_else(|| vector::cycles_after_issue(ball_class, funct, xs1, xs2))
        .unwrap_or_else(|| panic!("no goban BEMU ball latency implementation for ballClass={ball_class} funct7={funct}"))
}
