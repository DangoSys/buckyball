#[no_mangle]
pub extern "C" fn dma_ref_byte(address: u64) -> u32 {
    ((address ^ (address >> 8)) & 255) as u32
}
