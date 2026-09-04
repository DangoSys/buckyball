mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{MatrixCase, MatrixCmd};

thread_local! {
    static CURRENT: RefCell<Option<MatrixCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn smatmul_case_load(seed: u32, index: u32, bid: u32) {
    CURRENT.with(|current| *current.borrow_mut() = Some(casegen::gen_case(seed, index, bid)));
}

fn current_case<F: FnOnce(&MatrixCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| f(current.borrow().as_ref().expect("smatmul case not loaded")))
}

#[no_mangle]
pub extern "C" fn smatmul_case_num_commands() -> u32 {
    current_case(|case| case.commands.len() as u32)
}

#[no_mangle]
pub extern "C" fn smatmul_case_cmd(index: u32, out: *mut MatrixCmd) {
    assert!(!out.is_null(), "smatmul_case_cmd: null output");
    current_case(|case| unsafe {
        *out = *case
            .commands
            .get(index as usize)
            .expect("command index out of range");
    });
}

#[no_mangle]
pub extern "C" fn smatmul_case_bias_words() -> u32 {
    4
}

#[no_mangle]
pub extern "C" fn smatmul_case_a_words(block: u32) -> u32 {
    current_case(|case| (case.a[block as usize].len() / model::ROW_BYTES) as u32)
}

#[no_mangle]
pub extern "C" fn smatmul_case_b_words(block: u32) -> u32 {
    current_case(|case| (case.b[block as usize].len() / model::ROW_BYTES) as u32)
}

fn word_halves(data: &[u8], index: u32) -> (u64, u64) {
    let word = model::word(data, index as usize);
    (
        u64::from_le_bytes(word[0..8].try_into().unwrap()),
        u64::from_le_bytes(word[8..16].try_into().unwrap()),
    )
}

#[no_mangle]
pub extern "C" fn smatmul_case_bias_word_lo(index: u32) -> u64 {
    current_case(|case| word_halves(&case.bias, index).0)
}

#[no_mangle]
pub extern "C" fn smatmul_case_bias_word_hi(index: u32) -> u64 {
    current_case(|case| word_halves(&case.bias, index).1)
}

#[no_mangle]
pub extern "C" fn smatmul_case_a_word_lo(block: u32, index: u32) -> u64 {
    current_case(|case| word_halves(&case.a[block as usize], index).0)
}

#[no_mangle]
pub extern "C" fn smatmul_case_a_word_hi(block: u32, index: u32) -> u64 {
    current_case(|case| word_halves(&case.a[block as usize], index).1)
}

#[no_mangle]
pub extern "C" fn smatmul_case_b_word_lo(block: u32, index: u32) -> u64 {
    current_case(|case| word_halves(&case.b[block as usize], index).0)
}

#[no_mangle]
pub extern "C" fn smatmul_case_b_word_hi(block: u32, index: u32) -> u64 {
    current_case(|case| word_halves(&case.b[block as usize], index).1)
}

#[no_mangle]
pub extern "C" fn smatmul_case_num_writes() -> u32 {
    current_case(|case| case.writes.len() as u32)
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_addr(index: u32) -> u32 {
    current_case(|case| case.writes[index as usize].addr)
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_data_lo(index: u32) -> u64 {
    current_case(|case| {
        u64::from_le_bytes(case.writes[index as usize].data[0..8].try_into().unwrap())
    })
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_data_hi(index: u32) -> u64 {
    current_case(|case| {
        u64::from_le_bytes(case.writes[index as usize].data[8..16].try_into().unwrap())
    })
}
