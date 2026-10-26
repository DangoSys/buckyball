struct Memory { base: u64, bytes: Vec<u8>, initialized: Vec<bool> }

#[no_mangle]
pub extern "C" fn spm_create(base: u64, bytes: u32) -> *mut std::ffi::c_void {
    assert!(bytes > 0 && (base as u128 + bytes as u128) <= (1u128 << 64));
    Box::into_raw(Box::new(Memory {
        base, bytes: vec![0; bytes as usize], initialized: vec![false; bytes as usize],
    })).cast()
}

#[no_mangle]
pub unsafe extern "C" fn spm_destroy(handle: *mut std::ffi::c_void) {
    drop(unsafe { Box::from_raw(handle.cast::<Memory>()) });
}

// 0: success, 1: access error, 2: test attempted to compare uninitialized SRAM.
#[no_mangle]
pub unsafe extern "C" fn spm_access(
    handle: *mut std::ffi::c_void, address: u64, size: u32, write: u8,
    mask: u32, low: u64, high: u64, read_only: u8, out_low: *mut u64, out_high: *mut u64,
) -> u32 {
    let m = unsafe { &mut *handle.cast::<Memory>() };
    unsafe { *out_low = 0; *out_high = 0; }
    if size > 4 { return 1; }
    let count = 1usize << size;
    let Some(offset) = address.checked_sub(m.base) else { return 1; };
    let Some(end) = offset.checked_add(count as u64) else { return 1; };
    if address % count as u64 != 0 || end > m.bytes.len() as u64 { return 1; }
    if write != 0 && (read_only != 0 || mask >> count != 0) { return 1; }
    let offset = offset as usize;
    let input = (high as u128) << 64 | low as u128;
    if write != 0 {
        for i in 0..count {
            if mask & (1 << i) != 0 {
                m.bytes[offset+i] = (input >> (8*i)) as u8;
                m.initialized[offset+i] = true;
            }
        }
    } else {
        let mut value = 0u128;
        for i in 0..count {
            if !m.initialized[offset+i] { return 2; }
            value |= (m.bytes[offset+i] as u128) << (8*i);
        }
        unsafe { *out_low = value as u64; *out_high = (value >> 64) as u64; }
    }
    0
}
