mod casegen;
mod config;

use std::cell::RefCell;

use casegen::{Case, MxquantCmd, Step};

thread_local! {
    static CURRENT: RefCell<Option<Case>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn mxquant_num_cases() -> u32 {
    casegen::NUM_CASES
}

#[no_mangle]
pub extern "C" fn mxquant_case_load(index: u32) -> u32 {
    let case = casegen::gen_case(index, config::bank_entries());
    let steps = case.steps.len() as u32;
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    steps
}

fn current_step<F: FnOnce(&Step) -> R, R>(step: u32, f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => match case.steps.get(step as usize) {
            Some(s) => f(s),
            None => panic!("mxquant_case: step {step} out of range {}", case.steps.len()),
        },
        None => panic!("mxquant_case: no case loaded; call mxquant_case_load first"),
    })
}

fn idx(index: u32, limit: usize, what: &str) -> usize {
    if index as usize >= limit {
        panic!("mxquant_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn mxquant_case_cmd(step: u32, out_ptr: *mut MxquantCmd) {
    if out_ptr.is_null() {
        panic!("mxquant_case_cmd: null out_ptr");
    }
    current_step(step, |s| unsafe { *out_ptr = s.cmd });
}

#[no_mangle]
pub extern "C" fn mxquant_case_preload_count(step: u32) -> u32 {
    current_step(step, |s| s.preload.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxquant_case_preload_bank(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.preload[idx(i, s.preload.len(), "preload")].bank)
}

#[no_mangle]
pub extern "C" fn mxquant_case_preload_group(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.preload[idx(i, s.preload.len(), "preload")].group)
}

#[no_mangle]
pub extern "C" fn mxquant_case_preload_addr(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.preload[idx(i, s.preload.len(), "preload")].addr)
}

#[no_mangle]
pub extern "C" fn mxquant_case_preload_lo(step: u32, i: u32) -> u64 {
    current_step(step, |s| s.preload[idx(i, s.preload.len(), "preload")].data as u64)
}

#[no_mangle]
pub extern "C" fn mxquant_case_preload_hi(step: u32, i: u32) -> u64 {
    current_step(step, |s| (s.preload[idx(i, s.preload.len(), "preload")].data >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn mxquant_case_read_count(step: u32) -> u32 {
    current_step(step, |s| s.reads.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxquant_case_read_addr(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.reads[idx(i, s.reads.len(), "read")])
}

#[no_mangle]
pub extern "C" fn mxquant_case_write_count(step: u32) -> u32 {
    current_step(step, |s| s.writes.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxquant_case_write_addr(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.writes[idx(i, s.writes.len(), "write")].addr)
}

#[no_mangle]
pub extern "C" fn mxquant_case_write_mask(step: u32, i: u32) -> u32 {
    current_step(step, |s| u32::from(s.writes[idx(i, s.writes.len(), "write")].mask))
}

#[no_mangle]
pub extern "C" fn mxquant_case_write_lo(step: u32, i: u32) -> u64 {
    current_step(step, |s| s.writes[idx(i, s.writes.len(), "write")].data as u64)
}

#[no_mangle]
pub extern "C" fn mxquant_case_write_hi(step: u32, i: u32) -> u64 {
    current_step(step, |s| (s.writes[idx(i, s.writes.len(), "write")].data >> 64) as u64)
}
