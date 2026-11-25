use crate::cases::{gen_case, Case, Mxint8mmCmd, Step, NUM_CASES};
use std::cell::RefCell;

thread_local! {
    static CURRENT: RefCell<Option<Case>> = const { RefCell::new(None) };
}

fn current_step<R>(step: u32, f: impl FnOnce(&Step) -> R) -> R {
    CURRENT.with(|current| {
        let current = current.borrow();
        let case = current.as_ref().expect("mxint8mm_case: no case loaded");
        f(&case.steps[step as usize])
    })
}

#[no_mangle]
pub extern "C" fn mxint8mm_num_cases() -> u32 {
    NUM_CASES
}

#[no_mangle]
pub extern "C" fn mxint8mm_case_load(index: u32, bank_entries: u32, rob_entries: u32) -> u32 {
    let case = gen_case(index, bank_entries, rob_entries);
    let steps = case.steps.len() as u32;
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    steps
}

#[no_mangle]
pub extern "C" fn mxint8mm_case_cmd(step: u32, out: *mut Mxint8mmCmd) {
    current_step(step, |s| unsafe { *out = s.cmd });
}

#[no_mangle]
pub extern "C" fn mxint8mm_case_preload_count(step: u32) -> u32 {
    current_step(step, |s| s.preload.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxint8mm_case_preload(
    step: u32,
    i: u32,
    bank: *mut u32,
    group: *mut u32,
    addr: *mut u32,
    lo: *mut u64,
    hi: *mut u64,
) {
    current_step(step, |s| {
        let p = &s.preload[i as usize];
        unsafe {
            *bank = p.bank;
            *group = p.group;
            *addr = p.addr;
            *lo = p.data as u64;
            *hi = (p.data >> 64) as u64;
        }
    });
}

#[no_mangle]
pub extern "C" fn mxint8mm_case_write_count(step: u32) -> u32 {
    current_step(step, |s| s.writes.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxint8mm_case_write(step: u32, i: u32, addr: *mut u32, lo: *mut u64, hi: *mut u64) {
    current_step(step, |s| {
        let (at, data) = s.writes[i as usize];
        unsafe {
            *addr = at;
            *lo = data as u64;
            *hi = (data >> 64) as u64;
        }
    });
}

/// The cycles BEMU models for a MXINT8MM instruction (iter, rs2) on an array
/// of `rows` x `cols` columns of `lanes` code pairs a cycle.
#[no_mangle]
pub extern "C" fn mxint8mm_bemu_cycles(iter: u64, rs2: u64, rows: u64, cols: u64, lanes: u64) -> u64 {
    crate::latency::cycles(iter, rs2, rows, cols, 32 / lanes)
}
