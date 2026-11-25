const BALL_CLASS: &str = "examples.balls.intsoftmax.IntSoftmaxBall";

fn execute(class: &str, funct: u32, a: u64, b: u64, ctx: &mut bebop_bemu::ExecContext) -> Option<u64> {
    if class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("INTSOFTMAX") => Some(super::softmax::execute(a, b, ctx)),
        _ => None,
    }
}

fn cycles(class: &str, funct: u32, a: u64, b: u64, _ctx: &bebop_bemu::ExecContext) -> Option<u64> {
    if class != BALL_CLASS {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("INTSOFTMAX") => Some(super::softmax::cycles(a, b)),
        _ => None,
    }
}

#[bebop_bemu::ball::linkme::distributed_slice(bebop_bemu::BALLS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static BALL: bebop_bemu::Ball = bebop_bemu::Ball { execute, cycles };
