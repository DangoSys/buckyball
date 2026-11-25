// DPI interface of the MXINT8MM cases: load a case, then read its bank image,
// its commands and the C window each command must leave behind.
use crate::cases::{gen_case, Case, NUM_CASES};
use std::cell::RefCell;

thread_local! {
    static CURRENT: RefCell<Option<Case>> = const { RefCell::new(None) };
}

/// One bank row: (bank, group, row) and its 128 bits as four words.
#[repr(C)]
pub struct Row {
    pub bank: u32,
    pub group: u32,
    pub addr: u32,
    pub d0: u32,
    pub d1: u32,
    pub d2: u32,
    pub d3: u32,
}

#[repr(C)]
pub struct Cmd {
    pub iter: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub wr_read_bank: u32,
    pub op1_group: u32,
    pub op2_group: u32,
    pub wr_group: u32,
    pub features: u32,
}

fn row(key: (u32, u32, u32), data: u128) -> Row {
    Row {
        bank: key.0,
        group: key.1,
        addr: key.2,
        d0: data as u32,
        d1: (data >> 32) as u32,
        d2: (data >> 64) as u32,
        d3: (data >> 96) as u32,
    }
}

fn bank_entries() -> usize {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .expect("bank_entries missing from BB_VERIFY_CONFIG")
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}

fn with_case<R>(f: impl FnOnce(&Case) -> R) -> R {
    CURRENT.with(|current| f(current.borrow().as_ref().expect("mxint8matmul: no case loaded")))
}

#[no_mangle]
pub extern "C" fn mxint8matmul_bank_entries() -> u32 {
    bank_entries() as u32
}

#[no_mangle]
pub extern "C" fn mxint8matmul_num_cases() -> u32 {
    NUM_CASES
}

/// Load case `index`; return its number of commands.
#[no_mangle]
pub extern "C" fn mxint8matmul_case_load(index: u32) -> u32 {
    let case = gen_case(index, bank_entries());
    let steps = case.steps.len() as u32;
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    steps
}

#[no_mangle]
pub extern "C" fn mxint8matmul_case_preloads() -> u32 {
    with_case(|case| case.preload.len() as u32)
}

#[no_mangle]
pub extern "C" fn mxint8matmul_case_preload(i: u32, out: *mut Row) {
    with_case(|case| {
        let (key, data) = case.preload[i as usize];
        unsafe { *out = row(key, data) };
    })
}

#[no_mangle]
pub extern "C" fn mxint8matmul_case_cmd(step: u32, out: *mut Cmd) {
    with_case(|case| {
        let step = &case.steps[step as usize];
        let c = &step.cmd;
        let rs2 = c.rs2();
        unsafe {
            *out = Cmd {
                iter: c.k as u32,
                rs2_lo: rs2 as u32,
                rs2_hi: (rs2 >> 32) as u32,
                op1_bank: c.a.0,
                op2_bank: c.b.0,
                wr_bank: c.c.0,
                wr_read_bank: c.read_bank,
                op1_group: c.a.1,
                op2_group: c.b.1,
                wr_group: c.c.1,
                features: step.features,
            }
        };
    })
}

/// The expected write of C window row `offset` of command `step`.
#[no_mangle]
pub extern "C" fn mxint8matmul_case_expect(step: u32, offset: u32, out: *mut Row) {
    with_case(|case| {
        let step = &case.steps[step as usize];
        let c = &step.cmd;
        let key = (c.c.0, c.c.1, c.c_base as u32 + offset);
        unsafe { *out = row(key, step.expect[offset as usize]) };
    })
}

/// The cycles BEMU models for an MXINT8MM instruction (iter, rs2) on a
/// `rows` x `cols` array of `lanes`-lane PEs.
#[no_mangle]
pub extern "C" fn mxint8matmul_bemu_cycles(iter: u64, rs2: u64, rows: u64, cols: u64, lanes: u64) -> u64 {
    crate::latency::cycles(
        rs2 & 0x3ff,
        (rs2 >> 10) & 0x3ff,
        iter,
        rs2 & (1 << 62) != 0,
        rows,
        cols,
        lanes,
    )
}
