use bebop_bemu::ball::ExecContext;

pub fn execute_known(class: &str, funct: u32, xs1: u64, xs2: u64, ctx: &mut ExecContext) -> Option<u64> {
    if class != "examples.balls.fixed.FixedBall" {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("FIXED") => Some(super::operations::execute(xs1, xs2, ctx)),
        _ => None,
    }
}

pub fn cycles_after_issue(class: &str, funct: u32, xs1: u64, xs2: u64, ctx: &ExecContext) -> Option<u64> {
    if class != "examples.balls.fixed.FixedBall" {
        return None;
    }
    match bebop_bemu::ball::blink::mnemonic_for_funct(funct).as_deref() {
        Some("FIXED") => Some(super::operations::cycles(xs1, xs2, ctx)),
        _ => None,
    }
}
