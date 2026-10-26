mod casegen;

use std::cell::RefCell;

use casegen::{LutCase, LutCmd};

thread_local! {
    static CURRENT: RefCell<Option<LutCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn lut_case_load(index: u32) -> i32 {
    let case = casegen::gen_case(index);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&LutCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("lut_case: no case loaded; call lut_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn lut_case_cmd(out_ptr: *mut LutCmd) {
    if out_ptr.is_null() {
        panic!("lut_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn src_index(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("lut_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn lut_case_src_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.src_lo(src_index(word_index, case.cmd.num_src_words, "src_word_lo")))
}

#[no_mangle]
pub extern "C" fn lut_case_src_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.src_hi(src_index(word_index, case.cmd.num_src_words, "src_word_hi")))
}

#[no_mangle]
pub extern "C" fn lut_case_lut_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.lut_lo(src_index(word_index, case.cmd.num_lut_words, "lut_word_lo")))
}

#[no_mangle]
pub extern "C" fn lut_case_lut_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.lut_hi(src_index(word_index, case.cmd.num_lut_words, "lut_word_hi")))
}

#[no_mangle]
pub extern "C" fn lut_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.dst_lo(src_index(word_index, case.cmd.num_dst_words, "dst_word_lo")))
}

#[no_mangle]
pub extern "C" fn lut_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.dst_hi(src_index(word_index, case.cmd.num_dst_words, "dst_word_hi")))
}
