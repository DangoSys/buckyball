mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{VecMat16Case, VecMat16Cmd};

thread_local! {
    static CURRENT: RefCell<Option<VecMat16Case>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn vector_case_load(index: u32) -> i32 {
    let case = casegen::gen_case(index);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&VecMat16Case) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("vector_case: no case loaded; call vector_case_load first"),
    })
}

fn word(words: &[u128], index: u32, what: &str) -> u128 {
    match words.get(index as usize) {
        Some(word) => *word,
        None => panic!("vector_case_{what}: index {index} out of range {}", words.len()),
    }
}

#[no_mangle]
pub extern "C" fn vector_case_cmd(out_ptr: *mut VecMat16Cmd) {
    if out_ptr.is_null() {
        panic!("vector_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

#[no_mangle]
pub extern "C" fn vector_case_at_word_lo(index: u32) -> u64 {
    current_case(|case| word(&case.at, index, "at_word_lo") as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_at_word_hi(index: u32) -> u64 {
    current_case(|case| (word(&case.at, index, "at_word_hi") >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_b_word_lo(index: u32) -> u64 {
    current_case(|case| word(&case.b, index, "b_word_lo") as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_b_word_hi(index: u32) -> u64 {
    current_case(|case| (word(&case.b, index, "b_word_hi") >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_c_old_word_lo(index: u32) -> u64 {
    current_case(|case| word(&case.c_old, index, "c_old_word_lo") as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_c_old_word_hi(index: u32) -> u64 {
    current_case(|case| (word(&case.c_old, index, "c_old_word_hi") >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_c_new_word_lo(index: u32) -> u64 {
    current_case(|case| word(&case.c_new, index, "c_new_word_lo") as u64)
}

#[no_mangle]
pub extern "C" fn vector_case_c_new_word_hi(index: u32) -> u64 {
    current_case(|case| (word(&case.c_new, index, "c_new_word_hi") >> 64) as u64)
}
