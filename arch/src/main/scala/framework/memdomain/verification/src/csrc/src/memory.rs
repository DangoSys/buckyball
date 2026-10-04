//! Byte-addressed golden storage, independent of DUT state and completion timing.
use std::collections::BTreeMap;
#[derive(Default)]
pub struct Memory(BTreeMap<u64, u8>);
#[no_mangle]
pub extern "C" fn ack_ref_create() -> *mut Memory { Box::into_raw(Box::new(Memory::default())) }
#[no_mangle]
pub unsafe extern "C" fn ack_ref_destroy(m: *mut Memory) { drop(Box::from_raw(m)); }
#[no_mangle]
pub unsafe extern "C" fn ack_ref_write(m: *mut Memory, addr: u64, lo: u64, hi: u64, mask: u32) {
    let data = (lo as u128 | ((hi as u128) << 64)).to_le_bytes();
    for (i, byte) in data.into_iter().enumerate() { if mask & (1 << i) != 0 { (*m).0.insert(addr + i as u64, byte); } }
}
#[no_mangle]
pub unsafe extern "C" fn ack_ref_check(m: *mut Memory, addr: u64, lo: u64, hi: u64) -> u32 {
    let data = (lo as u128 | ((hi as u128) << 64)).to_le_bytes();
    u32::from(data.iter().enumerate().all(|(i, byte)| (*m).0.get(&(addr + i as u64)) == Some(byte)))
}
