//! The Ball's dispatch by mnemonic and its registration in BALLS.
const BALL_CLASS: &str = "examples.designs.piplus.balls.mxint8mm.Mxint8mmBall";

fn execute(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &mut bebop_bemu::ExecContext,
) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXINT8MM") => Some(super::matmul::execute(xs1, xs2, ctx)),
        _ => None,
    }
}

fn cycles(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &bebop_bemu::ExecContext,
) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("MXINT8MM") => Some(super::matmul::latency(xs1, xs2, ctx)),
        _ => None,
    }
}

#[bebop_bemu::ball::linkme::distributed_slice(bebop_bemu::BALLS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static BALL: bebop_bemu::Ball = bebop_bemu::Ball { execute, cycles };
