use bebop_bemu::ball::ExecContext;

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
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("F32ADD") => Some(super::add::execute(xs1, xs2, ctx)),
        _ => None,
    }
}
pub fn cycles_after_issue(
    ball_class: &str,
    funct: u32,
    xs1: u64,
    xs2: u64,
    _ctx: &bebop_bemu::ExecContext,
) -> Option<u64> {
    if ball_class != "examples.balls.f32add.F32AddBall" {
        return None;
    }
    match bebop_bemu::configs::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("F32ADD") => {
            use bebop_bemu::{
                configs::is_shared_vbank,
                thread::instructions::decode::{rs1_b0, rs1_b1, rs1_b2, rs1_iter},
            };
            let shared_read =
                is_shared_vbank(rs1_b0(xs1)) || (xs2 & 32 == 0 && is_shared_vbank(rs1_b1(xs1)));
            let shared_write = is_shared_vbank(rs1_b2(xs1));
            // Read handshake, registered result selection, add, and write ack.
            let row = 11 + 4 * u64::from(shared_read) + 4 * u64::from(shared_write);
            Some(8 + row * rs1_iter(xs1) + 3 * u64::from(shared_read))
        }
        _ => None,
    }
}
