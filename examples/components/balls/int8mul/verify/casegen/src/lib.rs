mod casegen;
mod config;

use std::cell::RefCell;

use casegen::{Int8MulCase, Int8MulCmd};

thread_local! {
    static CURRENT: RefCell<Option<Int8MulCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn int8mul_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid, config::bank_entries());
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&Int8MulCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("int8mul_case: no case loaded; call int8mul_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn int8mul_case_cmd(out_ptr: *mut Int8MulCmd) {
    if out_ptr.is_null() {
        panic!("int8mul_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn idx(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("int8mul_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn int8mul_case_gate_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.gate_lo(idx(word_index, case.cmd.num_gate_words, "gate_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_gate_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.gate_hi(idx(word_index, case.cmd.num_gate_words, "gate_word_hi")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_input_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.input_lo(idx(word_index, case.cmd.num_input_words, "input_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_input_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.input_hi(idx(word_index, case.cmd.num_input_words, "input_word_hi")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.dst_lo(idx(word_index, case.cmd.num_dst_words, "dst_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.dst_hi(idx(word_index, case.cmd.num_dst_words, "dst_word_hi")))
}
