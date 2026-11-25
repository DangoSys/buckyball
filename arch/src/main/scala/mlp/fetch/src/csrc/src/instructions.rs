fn byte(address: u64, version: u32) -> u8 {
    // Alternate C.NOP and ADDI; the 32-bit instructions start at PC=2 mod 6.
    let code = if version < 2 {
        [0x01, 0x00, 0x93, 0x00, (1 + version as u8) << 4, 0x00]
    } else {
        // Fetch transports arbitrary instruction bytes; execution/illegal-instruction decode is outside this IP.
        let bits = version.wrapping_mul(0x9e37_79b9);
        let compressed = ((bits as u16 & !3) | 1).to_le_bytes();
        let word = (bits.rotate_left(13) | 3).to_le_bytes();
        [compressed[0], compressed[1], word[0], word[1], word[2], word[3]]
    };
    code[(address % 6) as usize]
}
#[no_mangle]
pub extern "C" fn fetch_ref_word(address: u64, version: u32) -> u64 {
    let mut result = [0u8; 8];
    for (index, value) in result.iter_mut().enumerate() { *value = byte(address.wrapping_add(index as u64), version); }
    u64::from_le_bytes(result)
}
#[no_mangle]
pub extern "C" fn fetch_ref_group(address: u64, version: u32) -> u32 {
    let base = address - address % 4;
    u32::from_le_bytes(std::array::from_fn(|i| byte(base.wrapping_add(i as u64), version)))
}
#[no_mangle]
pub extern "C" fn fetch_ref_instruction(address: u64, version: u32) -> u32 {
    let first = u16::from_le_bytes([byte(address, version), byte(address.wrapping_add(1), version)]);
    if first & 3 != 3 { first as u32 } else {
        u32::from_le_bytes(std::array::from_fn(|i| byte(address.wrapping_add(i as u64), version)))
    }
}
