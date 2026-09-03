pub(crate) use crate::inst::decode;
pub(crate) use crate::inst::instruction;

use crate::inst::instruction::ExecContext;

mod quant;
mod model;

#[cfg(test)]
mod tests;

const BALL_CLASS: &str = "examples.balls.toint8.ToInt8Ball";

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
        Some("QUANT_F32_TO_I8") => Some(quant::exec_f32_to_i8(xs1, xs2, ctx)),
        Some("QUANT_I32_TO_I8") => Some(quant::exec_i32_to_i8(xs1, xs2, ctx)),
        Some(_) | None => None,
    }
}

pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> Option<u64> {
    if ball_class != BALL_CLASS {
        return None;
    }
    match crate::config::ball_domain::mnemonic_for_funct(funct).as_deref() {
        Some("QUANT_F32_TO_I8") => Some(quant::f32_to_i8_latency(xs1, xs2)),
        Some("QUANT_I32_TO_I8") => Some(quant::i32_to_i8_latency(xs1, xs2)),
        Some(_) | None => None,
    }
}
