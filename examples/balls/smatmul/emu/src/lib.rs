pub(crate) use crate::inst::{decode, instruction};

use crate::inst::instruction::ExecContext;

mod smatmul;

const BALL_CLASS: &str = "examples.balls.smatmul.SMatMulBall";

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
        Some("SMATMUL_OS") => Some(smatmul::exec_smatmul(false, xs1, xs2, ctx)),
        Some("SMATMUL_WS") => Some(smatmul::exec_smatmul(true, xs1, xs2, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("SMATMUL_OS") => Some(smatmul::latency(false, xs1, xs2)),
        Some("SMATMUL_WS") => Some(smatmul::latency(true, xs1, xs2)),
        _ => None,
    }
}
