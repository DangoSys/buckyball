mod fp16;
mod matmul;

use bebop_bemu::ball::ExecContext;

const CLASS: &str = "examples.balls.redmule.RedmuleBall";

pub fn execute_known(
    class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &mut ExecContext,
) -> Option<u64> {
    if class != CLASS {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("REDMULE_F16") => Some(matmul::execute(xs1, xs2, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(class: &str, funct: u32, _xs1: u64, _xs2: u64) -> Option<u64> {
    if class != CLASS {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("REDMULE_F16") => Some(1),
        _ => None,
    }
}

#[bebop_bemu::ball::linkme::distributed_slice(bebop_bemu::BALLS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static BALL: bebop_bemu::Ball = bebop_bemu::Ball { execute: execute_known, cycles: cycles_after_issue };
