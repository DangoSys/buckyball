use std::collections::BTreeMap;
struct Memory { bytes: BTreeMap<u64, u8> }
#[no_mangle]
pub extern "C" fn ddr_ref_create() -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(Memory { bytes: BTreeMap::new() })).cast()
}
#[no_mangle]
pub unsafe extern "C" fn ddr_ref_destroy(ptr: *mut std::ffi::c_void) {
    drop(Box::from_raw(ptr.cast::<Memory>()));
}
#[no_mangle]
pub unsafe extern "C" fn ddr_ref_program(ptr: *mut std::ffi::c_void, address: u64, data: *const u32) {
    let memory = &mut *ptr.cast::<Memory>();
    for byte in 0..64 {
        memory.bytes.insert(address + byte, ((*data.add(byte as usize / 4) >> ((byte % 4)*8)) & 255) as u8);
    }
}
#[no_mangle]
pub unsafe extern "C" fn ddr_ref_read(ptr: *mut std::ffi::c_void, address: u64, data: *mut u32) -> u32 {
    let memory = &*ptr.cast::<Memory>();
    for word in 0..16 { *data.add(word) = 0; }
    for byte in 0..64 {
        let Some(value) = memory.bytes.get(&(address+byte)) else { return 0; };
        *data.add(byte as usize / 4) |= (*value as u32) << ((byte%4)*8);
    }
    1
}
#[no_mangle]
pub unsafe extern "C" fn ddr_ref_write(ptr: *mut std::ffi::c_void, address: u64, data: *const u32, mask: u64) -> u32 {
    let memory = &mut *ptr.cast::<Memory>();
    if (0..64).any(|b| !memory.bytes.contains_key(&(address+b))) { return 0; }
    for byte in 0..64 {
        if mask & (1u64 << byte) != 0 {
            memory.bytes.insert(address+byte, ((*data.add(byte as usize/4) >> ((byte%4)*8)) & 255) as u8);
        }
    }
    1
}
