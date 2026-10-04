use bebop_memory::Memory;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr};
use std::path::Path;

const BASE: u64 = 0x8000_0000;
const BYTES: usize = 16usize << 30;

// External DDR starts zeroed. Storage is allocated one line at a time, using
// the maintained DDR reference for every program/read/write operation.
struct TileMemory {
    model: *mut c_void,
    initialized: RefCell<BTreeSet<u64>>,
}
impl TileMemory {
    fn new() -> Self {
        Self { model: crate::ddr_reference::ddr_ref_create(), initialized: RefCell::new(BTreeSet::new()) }
    }
    fn line(&self, address: u64) -> [u32; 16] {
        let mut data = [0u32; 16];
        if self.initialized.borrow_mut().insert(address) {
            unsafe { crate::ddr_reference::ddr_ref_program(self.model, address, data.as_ptr()); }
        }
        assert_eq!(unsafe { crate::ddr_reference::ddr_ref_read(self.model, address, data.as_mut_ptr()) }, 1,
            "DDR reference line must be initialized");
        data
    }
    fn check(offset: usize, bytes: usize) {
        assert!(offset.checked_add(bytes).is_some_and(|end| end <= BYTES), "Tile DDR access outside 16 GiB");
    }
}
impl Drop for TileMemory {
    fn drop(&mut self) { unsafe { crate::ddr_reference::ddr_ref_destroy(self.model); } }
}
impl Memory for TileMemory {
    fn len(&self) -> usize { BYTES }
    fn read_buffer(&self, offset: usize, output: &mut [u8]) {
        Self::check(offset, output.len());
        let mut cursor = 0;
        while cursor < output.len() {
            let address = BASE + (offset + cursor) as u64;
            let line = self.line(address & !63);
            let start = (address & 63) as usize;
            let count = (64 - start).min(output.len() - cursor);
            for i in 0..count {
                output[cursor + i] = (line[(start + i) / 4] >> (((start + i) % 4) * 8)) as u8;
            }
            cursor += count;
        }
    }
    fn write_buffer(&self, offset: usize, input: &[u8]) {
        Self::check(offset, input.len());
        let mut cursor = 0;
        while cursor < input.len() {
            let address = BASE + (offset + cursor) as u64;
            let mut line = self.line(address & !63);
            let start = (address & 63) as usize;
            let count = (64 - start).min(input.len() - cursor);
            for i in 0..count {
                let shift = ((start + i) % 4) * 8;
                line[(start + i) / 4] = (line[(start + i) / 4] & !(255 << shift)) | ((input[cursor + i] as u32) << shift);
            }
            unsafe { crate::ddr_reference::ddr_ref_program(self.model, address & !63, line.as_ptr()); }
            cursor += count;
        }
    }
    fn fill(&self, offset: usize, bytes: usize, value: u8) {
        Self::check(offset, bytes);
        let block = [value; 64];
        for cursor in (0..bytes).step_by(64) { self.write_buffer(offset + cursor, &block[..64.min(bytes - cursor)]); }
    }
}
thread_local! { static DDR: RefCell<Option<TileMemory>> = const { RefCell::new(None) }; }
thread_local! { static EXPECTED: RefCell<Option<String>> = const { RefCell::new(None) }; }

#[no_mangle]
pub unsafe extern "C" fn tile_memory_init(image_path: *const c_char, expected_path: *const c_char) -> u64 {
    let image = CStr::from_ptr(image_path).to_str().expect("Tile image path UTF-8");
    let expected = CStr::from_ptr(expected_path).to_str().expect("Tile expected bytes path UTF-8");
    assert!(!image.is_empty() && !expected.is_empty(), "Explicit Tile image and expected-byte paths are required");
    let memory = TileMemory::new();
    let entry = bebop_elf::load_elf(image, &memory, BASE, 0).expect("load Tile ELF").entry;
    assert!((BASE..BASE + BYTES as u64).contains(&entry), "Tile entry outside DDR");
    DDR.with(|ddr| { assert!(ddr.borrow().is_none(), "Tile peer initialized twice"); *ddr.borrow_mut() = Some(memory); });
    EXPECTED.with(|path| { *path.borrow_mut() = Some(expected.to_owned()); });
    bebop_rtl_trace::init_trace(Path::new("build/tile_system_trace"), bebop_rtl_trace::TraceConfig {
        itrace: true, mtrace: true, pmctrace: true, ctrace: true, banktrace: true,
    }).expect("initialize actual Tile RTL traces");
    println!("TILE_PEER_IMAGE image={image} entry={entry:#x} DDR=16GiB zero-initialized");
    entry
}

#[no_mangle]
pub extern "C" fn tile_peer_read64(address: u64) -> u64 {
    DDR.with(|ddr| {
        let memory = ddr.borrow();
        let mut data = [0; 8];
        memory.as_ref().expect("Tile peer not initialized").read_buffer(address.checked_sub(BASE).expect("AXI address below DDR") as usize, &mut data);
        u64::from_le_bytes(data)
    })
}
#[no_mangle]
pub extern "C" fn tile_peer_write128(address: u64, lo: u64, hi: u64, mask: u32) {
    assert_eq!(mask & !0xffff, 0, "AXI 128-bit write strobe width");
    assert_eq!(address & 15, 0, "AXI DDR write beat alignment");
    DDR.with(|ddr| {
        let memory = ddr.borrow();
        let memory = memory.as_ref().expect("Tile peer not initialized");
        let offset = address.checked_sub(BASE).expect("AXI address below DDR") as usize;
        TileMemory::check(offset, 16);
        let mut data = memory.line(address & !63);
        let incoming = (lo as u128 | ((hi as u128) << 64)).to_le_bytes();
        let start = (address & 63) as usize;
        for i in 0..16 {
            let shift = ((start + i) % 4) * 8;
            data[(start + i) / 4] = (data[(start + i) / 4] & !(255 << shift)) | ((incoming[i] as u32) << shift);
        }
        assert_eq!(unsafe { crate::ddr_reference::ddr_ref_write(memory.model, address & !63,
            data.as_ptr(), (mask as u64) << start) }, 1, "DDR AXI write to initialized reference line");
    });
}
#[no_mangle]
pub extern "C" fn tile_peer_finish() {
    let expected = EXPECTED.with(|path| path.borrow().as_ref().expect("Explicit Tile expected bytes path required").clone());
    let text = std::fs::read_to_string(&expected).expect("read expected Tile DDR output bytes");
    let mut compared = 0;
    DDR.with(|ddr| {
        let memory = ddr.borrow();
        let memory = memory.as_ref().expect("Tile peer not initialized");
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let fields: Vec<_> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 2, "Expected DDR row must contain address and hex bytes");
            let address = u64::from_str_radix(fields[0].trim_start_matches("0x"), 16).expect("expected DDR address");
            assert!(!fields[1].is_empty() && fields[1].len() % 2 == 0, "expected DDR byte string");
            let bytes: Vec<_> = (0..fields[1].len()).step_by(2).map(|i|
                u8::from_str_radix(&fields[1][i..i + 2], 16).expect("expected DDR byte")).collect();
            let mut actual = vec![0; bytes.len()];
            memory.read_buffer(address.checked_sub(BASE).expect("expected DDR address below RAM") as usize, &mut actual);
            assert_eq!(actual, bytes, "Actual Tile NPU DDR bytes differ at {address:#x}");
            compared += actual.len();
        }
    });
    assert!(compared > 0, "No independent Tile DDR output comparisons");
    println!("TILE_PEER_DDR_COMPARE bytes={compared} expected={expected}");
    bebop_rtl_trace::write_trace_summary(Path::new("build/tile_system_trace")).expect("write Tile RTL trace summary");
}
