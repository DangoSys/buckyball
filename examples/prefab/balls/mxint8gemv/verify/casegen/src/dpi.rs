// DPI interface of the BEMU cycle model: the test stages the A bank and the four
// B group banks an instruction reads, then asks for its cycles.
use std::cell::RefCell;

thread_local! {
    // Role 0 the A bank, 1..4 the B groups; 1024 rows of 16 bytes each.
    static BANKS: RefCell<[Vec<u8>; 5]> = RefCell::new(std::array::from_fn(|_| vec![0; 1024 * 16]));
}

/// Stages one 128-bit row (`lo` bits 63..0, `hi` bits 127..64) of `role`.
#[no_mangle]
pub extern "C" fn mxint8gemv_bemu_row(role: u32, row: u32, lo: u64, hi: u64) {
    BANKS.with(|banks| {
        let start = row as usize * 16;
        let bank = &mut banks.borrow_mut()[role as usize];
        bank[start..start + 8].copy_from_slice(&lo.to_le_bytes());
        bank[start + 8..start + 16].copy_from_slice(&hi.to_le_bytes());
    })
}

/// The cycles BEMU models for a MXINT8GEMV instruction (iter, rs2) on the staged
/// banks.
#[no_mangle]
pub extern "C" fn mxint8gemv_bemu_cycles(iter: u64, rs2: u64) -> u64 {
    BANKS.with(|banks| {
        let banks = banks.borrow();
        crate::latency::cycles(iter, rs2, &banks[0], [&banks[1], &banks[2], &banks[3], &banks[4]])
    })
}
