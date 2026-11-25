use std::ffi::c_void;

struct Memory {
    lines: Vec<[u8; 64]>,
    initialized: Vec<bool>,
}

#[no_mangle]
pub extern "C" fn chi_ref_create(lines: u32) -> *mut c_void {
    Box::into_raw(Box::new(Memory {
        lines: vec![[0; 64]; lines as usize],
        initialized: vec![false; lines as usize],
    })).cast()
}

#[no_mangle]
pub unsafe extern "C" fn chi_ref_destroy(model: *mut c_void) {
    drop(Box::from_raw(model.cast::<Memory>()));
}

// SRAM contents survive reset; an uninitialized line is never a golden zero.
#[no_mangle]
pub unsafe extern "C" fn chi_ref_write(
    model: *mut c_void, addr: u64, data: *const u32, mask: *const u32,
) -> u8 {
    let m = &mut *model.cast::<Memory>();
    assert_eq!(addr % 64, 0);
    let index = (addr / 64) as usize;
    if index >= m.lines.len() { return 1; }
    let data = std::slice::from_raw_parts(data, 16);
    let mask = std::slice::from_raw_parts(mask, 2);
    let full = mask.iter().all(|&word| word == u32::MAX);
    assert!(m.initialized[index] || full, "Partial write before initialization");
    for b in 0..64 {
        if mask[b / 32] & (1 << (b % 32)) != 0 {
            m.lines[index][b] = (data[b / 4] >> (8 * (b % 4))) as u8;
        }
    }
    m.initialized[index] = true;
    0
}

#[no_mangle]
pub unsafe extern "C" fn chi_ref_read(
    model: *mut c_void, addr: u64, data: *mut u32,
) -> u8 {
    let m = &*model.cast::<Memory>();
    assert_eq!(addr % 64, 0);
    let output = std::slice::from_raw_parts_mut(data, 16);
    output.fill(0);
    let index = (addr / 64) as usize;
    if index >= m.lines.len() { return 1; }
    assert!(m.initialized[index], "Read of uninitialized line");
    for (word, bytes) in output.iter_mut().zip(m.lines[index].chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    0
}
