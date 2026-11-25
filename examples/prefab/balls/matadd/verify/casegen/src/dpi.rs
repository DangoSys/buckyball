/// The cycles BEMU models for a MATADD instruction of `iter` rows in each of
/// `groups` bank groups.
#[no_mangle]
pub extern "C" fn matadd_bemu_cycles(iter: u64, groups: u64) -> u64 {
    crate::latency::cycles(iter, groups)
}
