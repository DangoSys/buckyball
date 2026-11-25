//! DPI entry point of the cycle model.

/// BEMU's cycles of one MXINT8UNPACK instruction.
#[no_mangle]
pub extern "C" fn mxint8unpack_bemu_cycles(iter: u64) -> u64 {
    crate::latency::cycles(iter)
}
