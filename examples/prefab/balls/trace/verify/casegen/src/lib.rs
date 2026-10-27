mod casegen;
mod model;

use std::cell::RefCell;

use model::{Command, Counters, Record};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TraceCmd {
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub rob_id: u32,
}

struct Case {
    cmds: Vec<Command>,
    counters: Counters,
    records: usize,
    errors: usize,
}

thread_local! {
    static CURRENT: RefCell<Option<Case>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn trace_case_load(index: u32) -> u32 {
    let cmds = casegen::gen_case(index);
    let count = cmds.len() as u32;
    CURRENT.with(|current| {
        *current.borrow_mut() = Some(Case { cmds, counters: Counters::default(), records: 0, errors: 0 })
    });
    count
}

fn with_case<F: FnOnce(&mut Case) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow_mut().as_mut() {
        Some(case) => f(case),
        None => panic!("trace_case: no case loaded; call trace_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn trace_case_cmd(index: u32, out_ptr: *mut TraceCmd) {
    if out_ptr.is_null() {
        panic!("trace_case_cmd: null out_ptr");
    }
    with_case(|case| {
        let cmd = match case.cmds.get(index as usize) {
            Some(cmd) => *cmd,
            None => panic!("trace_case_cmd: index {index} out of range {}", case.cmds.len()),
        };
        let rs2 = cmd.rs2();
        unsafe {
            *out_ptr = TraceCmd { rs2_lo: rs2 as u32, rs2_hi: (rs2 >> 32) as u32, rob_id: index % 16 };
        }
    });
}

// Records observed so far; -1 once any record mismatched the model.
#[no_mangle]
pub extern "C" fn trace_case_records() -> i32 {
    with_case(|case| if case.errors != 0 { -1 } else { case.records as i32 })
}

// The trace hook the RTL calls once per counter command.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dpi_ctrace(
    subcmd: u32,
    ctr_id: u32,
    tag_lo: u32,
    tag_hi: u32,
    elapsed_lo: u32,
    elapsed_hi: u32,
    cycle_lo: u32,
    cycle_hi: u32,
) {
    let got = Record {
        subcmd,
        ctr: ctr_id,
        tag: u64::from(tag_lo) | (u64::from(tag_hi) << 32),
        elapsed: u64::from(elapsed_lo) | (u64::from(elapsed_hi) << 32),
        cycle: u64::from(cycle_lo) | (u64::from(cycle_hi) << 32),
    };
    with_case(|case| {
        let index = case.records;
        case.records += 1;
        let cmd = match case.cmds.get(index) {
            Some(cmd) => *cmd,
            None => {
                eprintln!("[TRACE-REF] extra record {index}: {got:?}");
                case.errors += 1;
                return;
            }
        };
        match case.counters.expect(&cmd, got.cycle) {
            Ok(want) if want == got => {}
            Ok(want) => {
                eprintln!("[TRACE-REF] record {index} of {cmd:?}: got {got:?} expected {want:?}");
                case.errors += 1;
            }
            Err(reason) => {
                eprintln!("[TRACE-REF] record {index} of {cmd:?}: {reason}");
                case.errors += 1;
            }
        }
    });
}
