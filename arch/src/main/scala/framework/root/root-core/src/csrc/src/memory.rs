use std::ffi::{c_char, c_void, CStr};
const BASE: u64 = 0x8000_0000;
const BYTES: usize = 16 * 1024 * 1024;

#[no_mangle]
pub unsafe extern "C" fn core_ref_create(path: *const c_char) -> *mut c_void {
    let path = CStr::from_ptr(path).to_str().unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let mut memory = vec![0u8; BYTES];
    let mut address = BASE;
    for token in text.split_whitespace() {
        if let Some(value) = token.strip_prefix('@') { address = u64::from_str_radix(value, 16).unwrap(); }
        else {
            memory[usize::try_from(address - BASE).unwrap()] = u8::from_str_radix(token, 16).unwrap();
            address += 1;
        }
    }
    Box::into_raw(Box::new(memory)).cast()
}
#[no_mangle]
pub unsafe extern "C" fn core_ref_destroy(model: *mut c_void) {
    drop(Box::from_raw(model.cast::<Vec<u8>>()));
}
#[no_mangle]
pub unsafe extern "C" fn core_ref_read(model: *mut c_void, address: u64, output: *mut u32) {
    assert!(address % 64 == 0);
    let bytes = &*model.cast::<Vec<u8>>();
    let start = usize::try_from(address - BASE).unwrap();
    for i in 0..16 { *output.add(i) = u32::from_le_bytes(bytes[start + i*4..start + i*4 + 4].try_into().unwrap()); }
}
#[no_mangle]
pub unsafe extern "C" fn core_ref_write(model: *mut c_void, address: u64, data: *const u32, mask: *const u32) {
    assert!(address % 64 == 0);
    let bytes = &mut *model.cast::<Vec<u8>>();
    let start = usize::try_from(address - BASE).unwrap();
    let mask = u64::from(*mask) | (u64::from(*mask.add(1)) << 32);
    for i in 0..16 {
        let word = (*data.add(i)).to_le_bytes();
        for (b, value) in word.into_iter().enumerate() {
            if mask & (1u64 << (i*4+b)) != 0 { bytes[start + i*4 + b] = value; }
        }
    }
}
