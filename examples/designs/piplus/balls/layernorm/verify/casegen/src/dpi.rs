/// The cycles BEMU models for a LAYERNORM instruction (iter, rs2) on the 1024
/// bank rows (4096 words) of its input bank and its eps word.
#[no_mangle]
pub extern "C" fn layernorm_bemu_cycles(iter: u64, rs2: u64, words: *const u32, eps: u32) -> u64 {
    let input = unsafe { std::slice::from_raw_parts(words, 4096) };
    crate::latency::cycles(iter, rs2, input, eps)
}
