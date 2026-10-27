mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{Im2colCase, Im2colCmd};

thread_local! {
    static CURRENT: RefCell<Option<Im2colCase>> = const { RefCell::new(None) };
}

fn current_case<F: FnOnce(&Im2colCase) -> R, R>(f: F) -> R {
    CURRENT.with(|c| match c.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("im2col_case: no case loaded; call im2col_case_load first"),
    })
}

fn lo(row: &[u8; 16]) -> u64 {
    u64::from_le_bytes(row[..8].try_into().unwrap())
}

fn hi(row: &[u8; 16]) -> u64 {
    u64::from_le_bytes(row[8..].try_into().unwrap())
}

#[no_mangle]
pub extern "C" fn im2col_case_load(index: u32) -> i32 {
    let case = casegen::gen_case(index);
    CURRENT.with(|c| *c.borrow_mut() = Some(case));
    0
}

#[no_mangle]
pub extern "C" fn im2col_case_cmd(out_ptr: *mut Im2colCmd) {
    if out_ptr.is_null() {
        panic!("im2col_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe { *out_ptr = case.cmd });
}

#[no_mangle]
pub extern "C" fn im2col_case_src_group(i: u32) -> u32 {
    current_case(|case| case.src[i as usize].0 as u32)
}

#[no_mangle]
pub extern "C" fn im2col_case_src_addr(i: u32) -> u32 {
    current_case(|case| case.src[i as usize].1 as u32)
}

#[no_mangle]
pub extern "C" fn im2col_case_src_lo(i: u32) -> u64 {
    current_case(|case| lo(&case.src[i as usize].2))
}

#[no_mangle]
pub extern "C" fn im2col_case_src_hi(i: u32) -> u64 {
    current_case(|case| hi(&case.src[i as usize].2))
}

#[no_mangle]
pub extern "C" fn im2col_case_read_group(i: u32) -> u32 {
    current_case(|case| case.reads[i as usize].0 as u32)
}

#[no_mangle]
pub extern "C" fn im2col_case_read_addr(i: u32) -> u32 {
    current_case(|case| case.reads[i as usize].1 as u32)
}

#[no_mangle]
pub extern "C" fn im2col_case_dst_group(i: u32) -> u32 {
    current_case(|case| case.dst[i as usize].0 as u32)
}

#[no_mangle]
pub extern "C" fn im2col_case_dst_addr(i: u32) -> u32 {
    current_case(|case| case.dst[i as usize].1 as u32)
}

#[no_mangle]
pub extern "C" fn im2col_case_dst_lo(i: u32) -> u64 {
    current_case(|case| lo(&case.dst[i as usize].2))
}

#[no_mangle]
pub extern "C" fn im2col_case_dst_hi(i: u32) -> u64 {
    current_case(|case| hi(&case.dst[i as usize].2))
}
