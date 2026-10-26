use bebop_bemu::ball::ExecContext;

use super::im2col;

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
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("IM2COL") => Some(im2col::exec_im2col(xs1, xs2, ctx)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("IM2COL") => Some(im2col::latency(xs1, xs2, bebop_bemu::configs::ball_domain::instance_for_funct(funct).expect("Ball instance"))),
        Some(_) | None => None,
    }
}
