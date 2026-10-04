//! Physical DDR and CPU-visible architectural memory are distinct golden stores.
//! CPU dirty writes do not commit DDR. Only observed Home write ACKs or the mock DMA do.
use std::{collections::BTreeMap, ffi::c_void};
struct Memory {
    ddr: BTreeMap<u64, [u8; 64]>,
    visible: BTreeMap<u64, [u8; 64]>,
}
fn initial(address: u64) -> [u8; 64] {
    let mut line = [0; 64];
    for (i, byte) in line.iter_mut().enumerate() {
        *byte = ((address >> 6) as u8).wrapping_mul(19).wrapping_add((i as u8).wrapping_mul(13)) ^ 0x5a;
    }
    line
}
fn line(map: &BTreeMap<u64, [u8; 64]>, address: u64) -> [u8; 64] {
    *map.get(&address).unwrap_or(&initial(address))
}
fn write_word(map: &mut BTreeMap<u64, [u8; 64]>, address: u64, data: u64, mask: u32) {
    assert!(address < 1 << 44 && address & 7 == 0);
    let base = address & !63;
    let target = map.entry(base).or_insert_with(|| initial(base));
    for (i, byte) in data.to_le_bytes().iter().enumerate() {
        if mask & (1 << i) != 0 { target[(address & 63) as usize + i] = *byte; }
    }
}
unsafe fn memory<'a>(ptr: *mut c_void) -> &'a mut Memory { &mut *ptr.cast::<Memory>() }
#[no_mangle]
pub extern "C" fn cons_ref_create() -> *mut c_void {
    Box::into_raw(Box::new(Memory { ddr: BTreeMap::new(), visible: BTreeMap::new() })).cast()
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_destroy(ptr: *mut c_void) { drop(Box::from_raw(ptr.cast::<Memory>())); }
#[no_mangle]
pub unsafe extern "C" fn cons_ref_expected64(ptr: *mut c_void, address: u64) -> u64 {
    assert!(address & 7 == 0);
    let bytes = line(&memory(ptr).visible, address & !63);
    u64::from_le_bytes(bytes[(address & 63) as usize..(address & 63) as usize + 8].try_into().unwrap())
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_cpu_commit(ptr: *mut c_void, address: u64, data: u64, mask: u32) {
    write_word(&mut memory(ptr).visible, address, data, mask);
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_ddr_read(ptr: *mut c_void, address: u64, output: *mut u32) {
    assert!(address & 63 == 0 && address < 1 << 44);
    let bytes = line(&memory(ptr).ddr, address);
    for (i, word) in std::slice::from_raw_parts_mut(output, 16).iter_mut().enumerate() {
        *word = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_write_matches(ptr: *mut c_void, address: u64, data: *const u32, mask: *const u32) -> u32 {
    assert!(address & 63 == 0 && address < 1 << 44);
    let wanted = line(&memory(ptr).visible, address);
    let data = std::slice::from_raw_parts(data, 16);
    let mask = std::slice::from_raw_parts(mask, 2);
    u32::from((0..64).all(|i| mask[i / 32] & (1 << (i % 32)) == 0 || wanted[i] == (data[i / 4] >> (8 * (i % 4))) as u8))
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_ddr_commit(ptr: *mut c_void, address: u64, data: *const u32, mask: *const u32) {
    assert!(address & 63 == 0 && address < 1 << 44);
    let data = std::slice::from_raw_parts(data, 16);
    let mask = std::slice::from_raw_parts(mask, 2);
    let target = memory(ptr).ddr.entry(address).or_insert_with(|| initial(address));
    for (i, byte) in target.iter_mut().enumerate() {
        if mask[i / 32] & (1 << (i % 32)) != 0 { *byte = (data[i / 4] >> (8 * (i % 4))) as u8; }
    }
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_ddr_visible(ptr: *mut c_void, address: u64) -> u32 {
    assert!(address & 63 == 0);
    let m = memory(ptr);
    u32::from(line(&m.ddr, address) == line(&m.visible, address))
}
#[no_mangle]
pub unsafe extern "C" fn cons_ref_dma_write(ptr: *mut c_void, address: u64, data: u64, mask: u32) {
    let m = memory(ptr);
    write_word(&mut m.ddr, address, data, mask);
    write_word(&mut m.visible, address, data, mask);
}
