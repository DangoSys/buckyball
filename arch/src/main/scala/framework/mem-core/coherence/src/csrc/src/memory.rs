use std::collections::HashMap;
use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn coherence_ref_create() -> *mut c_void {
    Box::into_raw(Box::new(HashMap::<u64, [u32; 16]>::new())).cast()
}
#[no_mangle]
pub unsafe extern "C" fn coherence_ref_destroy(model: *mut c_void) {
    drop(Box::from_raw(model.cast::<HashMap<u64, [u32; 16]>>()));
}
#[no_mangle]
pub unsafe extern "C" fn coherence_ref_initial(addr: u64, out: *mut u32) {
    let out = std::slice::from_raw_parts_mut(out, 16);
    for (i, word) in out.iter_mut().enumerate() {
        *word = ((addr as u32) ^ ((addr >> 32) as u32)).rotate_left(i as u32)
            ^ (0x10203040u32.wrapping_mul(i as u32 + 1));
    }
}
#[no_mangle]
pub unsafe extern "C" fn coherence_ref_read(model: *mut c_void, addr: u64, out: *mut u32) {
    let memory = &mut *model.cast::<HashMap<u64, [u32; 16]>>();
    if let Some(data) = memory.get(&addr) {
        std::slice::from_raw_parts_mut(out, 16).copy_from_slice(data);
    } else {
        coherence_ref_initial(addr, out);
    }
}
#[no_mangle]
pub unsafe extern "C" fn coherence_ref_write(model: *mut c_void, addr: u64, data: *const u32) {
    let memory = &mut *model.cast::<HashMap<u64, [u32; 16]>>();
    memory.insert(
        addr,
        std::slice::from_raw_parts(data, 16).try_into().unwrap(),
    );
}
