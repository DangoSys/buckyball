mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{F32AddCase, F32AddCmd};

thread_local! {
    static CURRENT: RefCell<Option<F32AddCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn f32add_case_load(index: u32) -> i32 {
    let case = casegen::gen_case(index);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&F32AddCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("f32add_case: no case loaded; call f32add_case_load first"),
    })
}

fn idx(index: u32, case: &F32AddCase, what: &str) -> usize {
    if index >= case.cmd.rows {
        panic!("f32add_case_{what}: index {index} out of range {}", case.cmd.rows);
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn f32add_case_cmd(out_ptr: *mut F32AddCmd) {
    if out_ptr.is_null() {
        panic!("f32add_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

#[no_mangle]
pub extern "C" fn f32add_case_a_word_lo(index: u32) -> u64 {
    current_case(|case| case.a[idx(index, case, "a_word_lo")] as u64)
}

#[no_mangle]
pub extern "C" fn f32add_case_a_word_hi(index: u32) -> u64 {
    current_case(|case| (case.a[idx(index, case, "a_word_hi")] >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn f32add_case_b_word_lo(index: u32) -> u64 {
    current_case(|case| case.b[idx(index, case, "b_word_lo")] as u64)
}

#[no_mangle]
pub extern "C" fn f32add_case_b_word_hi(index: u32) -> u64 {
    current_case(|case| (case.b[idx(index, case, "b_word_hi")] >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn f32add_case_c_word_lo(index: u32) -> u64 {
    current_case(|case| case.c[idx(index, case, "c_word_lo")] as u64)
}

#[no_mangle]
pub extern "C" fn f32add_case_c_word_hi(index: u32) -> u64 {
    current_case(|case| (case.c[idx(index, case, "c_word_hi")] >> 64) as u64)
}
