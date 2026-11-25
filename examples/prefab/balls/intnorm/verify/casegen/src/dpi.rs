/// The cycles BEMU models for an INTNORM instruction (iter, rs2).
#[no_mangle]
pub extern "C" fn intnorm_bemu_cycles(iter: u64, rs2: u64) -> u64 {
    crate::latency::cycles(iter, rs2)
}
