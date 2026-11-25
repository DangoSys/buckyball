/// The cycles BEMU models for a FIXED instruction (iter, rs2) whose A and B
/// banks hold `a` and `b`: 4096 I32 words each, four per row.
#[no_mangle]
pub extern "C" fn fixed_bemu_cycles(iter: u64, rs2: u64, a: *const i32, b: *const i32) -> u64 {
    let a = unsafe { std::slice::from_raw_parts(a, 4096) };
    let b = unsafe { std::slice::from_raw_parts(b, 4096) };
    crate::latency::cycles(iter, rs2, a, b)
}
