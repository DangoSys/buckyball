mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{ReluCase, ReluCmd};

thread_local! {
    static CURRENT: RefCell<Option<ReluCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn relu_case_load(index: u32) -> i32 {
    let case = casegen::gen_case(index);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&ReluCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("relu_case: no case loaded; call relu_case_load first"),
    })
}

fn idx(index: u32, case: &ReluCase, what: &str) -> usize {
    if index >= case.cmd.num_words {
        panic!("relu_case_{what}: index {index} out of range {}", case.cmd.num_words);
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn relu_case_cmd(out_ptr: *mut ReluCmd) {
    if out_ptr.is_null() {
        panic!("relu_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

#[no_mangle]
pub extern "C" fn relu_case_row(index: u32) -> u32 {
    current_case(|case| case.rows[idx(index, case, "row")])
}

#[no_mangle]
pub extern "C" fn relu_case_src_word_lo(index: u32) -> u64 {
    current_case(|case| case.src[idx(index, case, "src_word_lo")] as u64)
}

#[no_mangle]
pub extern "C" fn relu_case_src_word_hi(index: u32) -> u64 {
    current_case(|case| (case.src[idx(index, case, "src_word_hi")] >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn relu_case_dst_word_lo(index: u32) -> u64 {
    current_case(|case| case.dst[idx(index, case, "dst_word_lo")] as u64)
}

#[no_mangle]
pub extern "C" fn relu_case_dst_word_hi(index: u32) -> u64 {
    current_case(|case| (case.dst[idx(index, case, "dst_word_hi")] >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn relu_bank_entries() -> u32 {
    model::bank_entries()
}
