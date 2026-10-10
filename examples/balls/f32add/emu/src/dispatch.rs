use crate::inst::instruction::ExecContext;

pub fn execute_known(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    ctx: &mut ExecContext,
) -> Option<u64> {
    if ball_class != "examples.balls.f32add.F32AddBall" {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("F32ADD") => Some(super::add::execute(xs1, xs2, ctx)),
        _ => None,
    }
}
pub fn cycles_after_issue(ball_class: &str, funct: u32, _xs1: u64, _xs2: u64) -> Option<u64> {
    if ball_class != "examples.balls.f32add.F32AddBall" {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("F32ADD") => Some(1),
        _ => None,
    }
}
