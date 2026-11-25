use crate::casegen;
use crate::config;
use std::cell::RefCell;

use crate::casegen::{Case, Mxint8quantCmd, Step};

thread_local! {
    static CURRENT: RefCell<Option<Case>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn mxint8quant_num_cases() -> u32 {
    casegen::NUM_CASES
}

#[no_mangle]
pub extern "C" fn mxint8quant_num_events() -> u32 {
    casegen::EVENTS as u32
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_load(index: u32) -> u32 {
    let case = casegen::gen_case(index, config::bank_entries());
    let steps = case.steps.len() as u32;
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    steps
}

fn current_step<F: FnOnce(&Step) -> R, R>(step: u32, f: F) -> R {
    CURRENT.with(|current| f(&current.borrow().as_ref().expect("mxint8quant: no case loaded").steps[step as usize]))
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_cmd(step: u32, out_ptr: *mut Mxint8quantCmd) {
    current_step(step, |s| unsafe { *out_ptr = s.cmd });
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_hold(step: u32) -> u32 {
    current_step(step, |s| s.hold)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_event(step: u32, event: u32) -> u32 {
    current_step(step, |s| s.events[event as usize])
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_preload_count(step: u32) -> u32 {
    current_step(step, |s| s.preload.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_preload_addr(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.preload[i as usize].addr)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_preload_lo(step: u32, i: u32) -> u64 {
    current_step(step, |s| s.preload[i as usize].data as u64)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_preload_hi(step: u32, i: u32) -> u64 {
    current_step(step, |s| (s.preload[i as usize].data >> 64) as u64)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_write_count(step: u32) -> u32 {
    current_step(step, |s| s.writes.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_write_addr(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.writes[i as usize].addr)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_write_kind(step: u32, i: u32) -> u32 {
    current_step(step, |s| s.writes[i as usize].kind)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_write_lo(step: u32, i: u32) -> u64 {
    current_step(step, |s| s.writes[i as usize].data as u64)
}

#[no_mangle]
pub extern "C" fn mxint8quant_case_write_hi(step: u32, i: u32) -> u64 {
    current_step(step, |s| (s.writes[i as usize].data >> 64) as u64)
}

/// The cycles BEMU models for a MXINT8QUANT instruction (iter, rs2).
#[no_mangle]
pub extern "C" fn mxint8quant_bemu_cycles(iter: u64, rs2: u64) -> u64 {
    crate::latency::cycles(iter, rs2)
}
