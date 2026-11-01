use super::quant;
use bebop_bemu::ball::ExecContext;

const BALL_CLASS: &str = "examples.balls.mxquant.MxquantBall";

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
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXQUANT") => Some(quant::execute(xs1, xs2, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    _xs2: u64,
    _ctx: &bebop_bemu::ExecContext,
) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXQUANT") => {
            Some(8 + 85 * bebop_bemu::thread::instructions::decode::rs1_iter(xs1) / 32)
        }
        _ => None,
    }
}
