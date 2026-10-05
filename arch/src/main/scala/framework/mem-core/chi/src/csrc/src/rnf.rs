use std::{collections::BTreeMap, ffi::c_void};

struct Memory {
    limit: u64,
    lines: BTreeMap<u64, [u8; 64]>,
}

#[no_mangle]
pub extern "C" fn rnf_ref_create(address_bits: u32) -> *mut c_void {
    assert!((44..=52).contains(&address_bits));
    Box::into_raw(Box::new(Memory {
        limit: 1u64 << address_bits,
        lines: BTreeMap::new(),
    })).cast()
}

#[no_mangle]
pub unsafe extern "C" fn rnf_ref_destroy(model: *mut c_void) {
    drop(Box::from_raw(model.cast::<Memory>()));
}

#[no_mangle]
pub unsafe extern "C" fn rnf_ref_write(
    model: *mut c_void, addr: u64, data: *const u32, mask: *const u32,
) -> u8 {
    let memory = &mut *model.cast::<Memory>();
    assert_eq!(addr % 64, 0);
    if addr >= memory.limit { return 1; }
    let data = std::slice::from_raw_parts(data, 16);
    let mask = std::slice::from_raw_parts(mask, 2);
    let full = mask.iter().all(|&word| word == u32::MAX);
    assert!(memory.lines.contains_key(&addr) || full, "Partial write before initialization");
    let line = memory.lines.entry(addr).or_insert([0; 64]);
    for b in 0..64 {
        if mask[b / 32] & (1 << (b % 32)) != 0 {
            line[b] = (data[b / 4] >> (8 * (b % 4))) as u8;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn rnf_ref_read(
    model: *mut c_void, addr: u64, data: *mut u32,
) -> u8 {
    let memory = &*model.cast::<Memory>();
    assert_eq!(addr % 64, 0);
    let output = std::slice::from_raw_parts_mut(data, 16);
    output.fill(0);
    if addr >= memory.limit { return 1; }
    let line = memory.lines.get(&addr).expect("Read of uninitialized line");
    for (word, bytes) in output.iter_mut().zip(line.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn rnf_ref_access(
    model: *mut c_void, addr: u64, write: u8, operand: u64,
    atomic: u32, word: u8, mask: u8, sc_success: u8, failed: u8,
) -> u64 {
    let memory = &mut *model.cast::<Memory>();
    assert!(atomic <= 12);
    assert!(word == 0 || (1..=11).contains(&atomic));
    assert_eq!(addr % if word != 0 { 4 } else { 8 }, 0);
    assert!(atomic == 0 || (write == 0 && mask == 255));
    if failed != 0 { return 0; }
    let line = memory.lines.get_mut(&(addr & !63)).expect("CPU access to uninitialized line");
    let offset = (addr & 63) as usize;
    let (old, update) = if word != 0 {
        let lhs = u32::from_le_bytes(line[offset..offset + 4].try_into().unwrap());
        let rhs = operand as u32;
        let result = match atomic {
            2 => lhs.wrapping_add(rhs),
            3 => lhs ^ rhs,
            4 => lhs & rhs,
            5 => lhs | rhs,
            6 => if (lhs as i32) < (rhs as i32) { lhs } else { rhs },
            7 => if (lhs as i32) > (rhs as i32) { lhs } else { rhs },
            8 => lhs.min(rhs),
            9 => lhs.max(rhs),
            _ => rhs,
        };
        ((lhs as i32 as i64) as u64, result as u64)
    } else {
        let lhs = u64::from_le_bytes(line[offset..offset + 8].try_into().unwrap());
        let rhs = operand;
        let result = match atomic {
            2 => lhs.wrapping_add(rhs),
            3 => lhs ^ rhs,
            4 => lhs & rhs,
            5 => lhs | rhs,
            6 => if (lhs as i64) < (rhs as i64) { lhs } else { rhs },
            7 => if (lhs as i64) > (rhs as i64) { lhs } else { rhs },
            8 => lhs.min(rhs),
            9 => lhs.max(rhs),
            _ => rhs,
        };
        (lhs, result)
    };
    if write != 0 || (1..=9).contains(&atomic) || (atomic == 11 && sc_success != 0) {
        for b in 0..if word != 0 { 4 } else { 8 } {
            if word != 0 || mask & (1 << b) != 0 {
                line[offset + b] = (update >> (8 * b)) as u8;
            }
        }
    }
    match atomic { 11 => u64::from(sc_success == 0), 12 => 0, _ => old }
}
