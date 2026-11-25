//! DPI entry points: the SV test loads one case and reads its fields and bank rows.
use crate::cases::{gen_case, Case, NUM_CASES};
use crate::reference::{row_stride, EVENTS};
use std::cell::RefCell;

thread_local! {
    static CASE: RefCell<Option<Case>> = const { RefCell::new(None) };
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
    CASE.with(|case| f(case.borrow().as_ref().expect("no case loaded")))
}

#[no_mangle]
pub extern "C" fn mxint8act_num_cases() -> u32 {
    NUM_CASES
}

#[no_mangle]
pub extern "C" fn mxint8act_num_events() -> u32 {
    EVENTS as u32
}

/// Loads case `index`; returns the bank depth in rows.
#[no_mangle]
pub extern "C" fn mxint8act_case_load(index: u32) -> u32 {
    let entries = bank_entries();
    let case = gen_case(index, entries);
    CASE.with(|current| *current.borrow_mut() = Some(case));
    entries as u32
}

/// 0 rows, 1 K, 2 gated, 3 input base, 4 output base, 5 table base,
/// 6 bank rows per logical row, 7 response hold cycles.
#[no_mangle]
pub extern "C" fn mxint8act_case_field(field: u32) -> u32 {
    with_case(|c| {
        [
            c.rows,
            c.k,
            usize::from(c.gated),
            c.in_base,
            c.out_base,
            c.table_base,
            row_stride(c.k),
            c.hold as usize,
        ][field as usize] as u32
    })
}

/// Half `high` of bank row `row` of image `role`: 0 input, 1 table,
/// 2 output before, 3 expected output.
#[no_mangle]
pub extern "C" fn mxint8act_case_word(role: u32, row: u32, high: u32) -> u64 {
    with_case(|c| {
        let at = row as usize * 16 + high as usize * 8;
        u64::from_le_bytes(c.banks[role as usize][at..at + 8].try_into().unwrap())
    })
}

/// How often the case exercises semantic path `kind` (see reference.rs).
#[no_mangle]
pub extern "C" fn mxint8act_case_events(kind: u32) -> u32 {
    with_case(|c| c.events[kind as usize])
}

/// BEMU's cycles of one MXINT8ACT instruction; `words` is the input bank,
/// two words per row, low half first.
#[no_mangle]
pub extern "C" fn mxint8act_bemu_cycles(iter: u64, rs2: u64, words: *const u64, entries: u32) -> u64 {
    let words = unsafe { std::slice::from_raw_parts(words, 2 * entries as usize) };
    let input: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    crate::latency::cycles(iter, rs2, &input)
}
