mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{TransposeCase, TransposeCmd, MAX_WORDS};

thread_local! {
    static CURRENT: RefCell<Option<TransposeCase>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn transpose_case_load(seed: u32, index: u32) -> i32 {
    let case = casegen::gen_case(seed, index);
    let nsrc = case.cmd.num_src_words as usize;
    let ndst = case.cmd.num_dst_words as usize;
    if nsrc > MAX_WORDS || ndst > MAX_WORDS {
        panic!("transpose_case_load: word count out of range src={nsrc} dst={ndst}");
    }
    CURRENT.with(|c| *c.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&TransposeCase) -> R, R>(f: F) -> R {
    CURRENT.with(|c| match c.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("transpose_case: no case loaded; call transpose_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn transpose_case_cmd(out_ptr: *mut TransposeCmd) {
    current_case(|case| unsafe {
        if out_ptr.is_null() {
            panic!("transpose_case_cmd: null out_ptr");
        }
        *out_ptr = case.cmd;
    });
}

#[no_mangle]
pub extern "C" fn transpose_case_src_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_src_word_lo: word_index out of range");
    }
    current_case(|case| case.src_word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn transpose_case_src_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_src_word_hi: word_index out of range");
    }
    current_case(|case| case.src_word_hi(word_index as usize))
}

#[no_mangle]
pub extern "C" fn transpose_case_dst_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_dst_word_lo: word_index out of range");
    }
    current_case(|case| case.dst_word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn transpose_case_dst_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_dst_word_hi: word_index out of range");
    }
    current_case(|case| case.dst_word_hi(word_index as usize))
}
