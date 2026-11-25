struct Bank {
    words: Vec<u32>,
    initialized: Vec<u8>,
}

impl Bank {
    fn access(&mut self, addr: usize, write: bool, data: u32, mask: u8) -> u32 {
        if write {
            for byte in 0..4 {
                if mask & (1 << byte) != 0 {
                    let bits = 0xff << (byte * 8);
                    self.words[addr] = (self.words[addr] & !bits) | (data & bits);
                }
            }
            self.initialized[addr] |= mask;
            0
        } else {
            assert_eq!(
                self.initialized[addr], 0xf,
                "read of uninitialized bank word"
            );
            self.words[addr]
        }
    }
}

#[no_mangle]
pub extern "C" fn bank_ref_create(entries: u32) -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(Bank {
        words: vec![0; entries as usize],
        initialized: vec![0; entries as usize],
    }))
    .cast()
}

#[no_mangle]
pub unsafe extern "C" fn bank_ref_destroy(model: *mut std::ffi::c_void) {
    drop(Box::from_raw(model.cast::<Bank>()));
}

#[no_mangle]
pub unsafe extern "C" fn bank_ref_access(
    model: *mut std::ffi::c_void,
    addr: u32,
    write: u8,
    data: u32,
    mask: u8,
) -> u32 {
    (*model.cast::<Bank>()).access(addr as usize, write != 0, data, mask)
}
