use bebop_bemu::ball::ExecContext;

pub fn execute_known(ball_class: &str, funct: u32, xs1: u64, xs2: u64, ctx: &mut ExecContext) -> Option<u64> {
    if ball_class != "examples.balls.f32add.F32AddBall" {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("F32ADD") => Some(super::add::execute(xs1, xs2, ctx)),
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
    if ball_class != "examples.balls.f32add.F32AddBall" {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("F32ADD") => Some(super::latency::cycles(bebop_bemu::ball::blink::rs1_iter(xs1))),
        _ => None,
    }
}
