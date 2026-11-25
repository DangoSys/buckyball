#[no_mangle]
pub extern "C" fn gemmini_casegen_present() -> u32 {
    1
}

/// The cycles BEMU models for a BLink command (iter, rs2) under the
/// configuration (ws, a_transpose, b_transpose) the last GEMMINI_CONFIG set.
#[no_mangle]
pub extern "C" fn gemmini_bemu_cycles(iter: u64, rs2: u64, ws: u8, a_transpose: u8, b_transpose: u8) -> u64 {
    crate::latency::cycles(iter, rs2, ws != 0, a_transpose != 0, b_transpose != 0)
}

/// The cycles BEMU models for a loop configuration on the expansion port.
#[no_mangle]
pub extern "C" fn gemmini_bemu_loop_config_cycles() -> u64 {
    crate::latency::loop_config_cycles()
}

/// The cycles BEMU models for GEMMINI_LOOP_WS with these bounds.
#[no_mangle]
pub extern "C" fn gemmini_bemu_loop_ws_cycles(max_i: u64, max_j: u64, max_k: u64) -> u64 {
    crate::latency::loop_ws_cycles(max_i, max_j, max_k)
}

/// The cycles BEMU models for GEMMINI_LOOP_CONV_WS with these dimensions.
#[no_mangle]
pub extern "C" fn gemmini_bemu_loop_conv_ws_cycles(
    batch: u64,
    out_dim: u64,
    out_channels: u64,
    in_channels: u64,
    kernel_dim: u64,
) -> u64 {
    crate::latency::loop_conv_ws_cycles(batch, out_dim, out_channels, in_channels, kernel_dim)
}
