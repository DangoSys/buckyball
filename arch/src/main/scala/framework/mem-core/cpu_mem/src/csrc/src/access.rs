#[no_mangle]
pub unsafe extern "C" fn cpu_mem_ref_prepare(
    addr: u64, size: u32, write: u32, data: u64, atomic: u32, cacheable: u32, normal: u32,
    address_bits: u32, target: *mut u32, misaligned: *mut u32, access_fault: *mut u32,
    bus_addr: *mut u64, bus_data: *mut u64, bus_mask: *mut u32, atomic_word: *mut u32,
) {
    assert!(size <= 3 && atomic <= 11 && (atomic == 0 || (write == 0 && size >= 2)));
    let width = 1usize << size;
    let bad_align = addr % width as u64 != 0;
    let bad_access = addr >= (1u64 << address_bits) || (normal == 0 && atomic != 0);
    *misaligned = bad_align as u32;
    *access_fault = (!bad_align && bad_access) as u32;
    *target = if bad_align || bad_access { 0 } else if cacheable != 0 { 1 } else { 2 };
    *bus_addr = if cacheable != 0 && atomic == 0 { addr - addr % 8 } else { addr };
    *atomic_word = (atomic != 0 && width == 4) as u32;
    if atomic != 0 { *bus_data = data; *bus_mask = 255; }
    else if cacheable == 0 { *bus_data = data; *bus_mask = 0; }
    else {
        let offset = (addr % 8) as usize;
        let input = data.to_le_bytes();
        let mut lanes = [0u8; 8];
        // Datapath outside the write mask is irrelevant; retain all shifted source bytes.
        for lane in offset..8 { lanes[lane] = input[lane - offset]; }
        *bus_data = u64::from_le_bytes(lanes);
        *bus_mask = if write != 0 && !bad_align {
            (offset..offset + width).fold(0, |m, b| m | (1 << b))
        } else { 0 };
    }
}

#[no_mangle]
pub extern "C" fn cpu_mem_ref_result(
    addr: u64, size: u32, write: u32, signed: u32, atomic: u32,
    cacheable: u32, raw: u64, error: u32,
) -> u64 {
    if write != 0 || error != 0 { return 0; }
    if atomic != 0 { return raw; } // Cache or centralized normal RAM returns architectural Word atomic values.
    let bytes = raw.to_le_bytes();
    let start = if cacheable != 0 { (addr % 8) as usize } else { 0 };
    let length = 1usize << size;
    let mut result = [0u8; 8];
    result[..length].copy_from_slice(&bytes[start..start + length]);
    if signed != 0 && result[length - 1] & 0x80 != 0 { result[length..].fill(255); }
    u64::from_le_bytes(result)
}
