mod casegen;
#[path = "../../../emu/src/model.rs"]
mod model;

use std::cell::RefCell;

use casegen::{ToInt8Case, ToInt8Cmd, MAX_DST, MAX_SCALE, MAX_SRC};

thread_local! {
    static CURRENT: RefCell<Option<ToInt8Case>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn toint8_case_load(index: u32) -> i32 {
    let case = casegen::gen_case(index);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&ToInt8Case) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("toint8_case: no case loaded; call toint8_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_cmd(out_ptr: *mut ToInt8Cmd) {
    if out_ptr.is_null() {
        panic!("toint8_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn src_index(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("toint8_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn toint8_case_src_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_src_words, "src_word_lo");
        if i >= MAX_SRC {
            panic!("toint8_case_src_word_lo: word_index out of range");
        }
        case.src_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_src_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_src_words, "src_word_hi");
        if i >= MAX_SRC {
            panic!("toint8_case_src_word_hi: word_index out of range");
        }
        case.src_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_scale_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_scale_words, "scale_word_lo");
        if i >= MAX_SCALE {
            panic!("toint8_case_scale_word_lo: word_index out of range");
        }
        case.scale_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_scale_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_scale_words, "scale_word_hi");
        if i >= MAX_SCALE {
            panic!("toint8_case_scale_word_hi: word_index out of range");
        }
        case.scale_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_dst_words, "dst_word_lo");
        if i >= MAX_DST {
            panic!("toint8_case_dst_word_lo: word_index out of range");
        }
        case.dst_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_dst_words, "dst_word_hi");
        if i >= MAX_DST {
            panic!("toint8_case_dst_word_hi: word_index out of range");
        }
        case.dst_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_dst_addr(word_index: u32) -> u32 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_dst_words, "dst_addr");
        if i >= MAX_DST {
            panic!("toint8_case_dst_addr: word_index out of range");
        }
        case.dst_addr[i]
    })
}
