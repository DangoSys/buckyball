#[path = "../../../../../../mem-core/spm/src/csrc/src/lib.rs"]
mod storage;

#[no_mangle]
pub extern "C" fn ant_arithmetic(a: u64, b: u64, operation: u32, word: bool) -> u64 {
    let (a, b) = if word { (a as u32 as u64, b as u32 as u64) } else { (a, b) };
    let signed = |x: u64| if word { x as u32 as i32 as i64 } else { x as i64 };
    let value = match operation {
        0 => a.wrapping_mul(b),
        1 => (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64,
        2 => (((a as i64 as i128) * (b as i128)) >> 64) as u64,
        3 => (((a as u128) * (b as u128)) >> 64) as u64,
        4 => if b == 0 { u64::MAX } else { signed(a).wrapping_div(signed(b)) as u64 },
        5 => if b == 0 { u64::MAX } else { a / b },
        6 => if b == 0 { a } else { signed(a).wrapping_rem(signed(b)) as u64 },
        7 => if b == 0 { a } else { a % b },
        _ => unreachable!(),
    };
    if word { value as u32 as i32 as i64 as u64 } else { value }
}
