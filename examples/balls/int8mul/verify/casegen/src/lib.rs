mod casegen;

use std::cell::RefCell;

use casegen::{Int8MulCase, Int8MulCmd};

thread_local! {
    static CURRENT: RefCell<Option<Int8MulCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn int8mul_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&Int8MulCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("int8mul_case: no case loaded; call int8mul_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn int8mul_case_cmd(out_ptr: *mut Int8MulCmd) {
    if out_ptr.is_null() {
        panic!("int8mul_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn idx(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("int8mul_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn int8mul_case_gate_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.gate_lo(idx(word_index, case.cmd.num_gate_words, "gate_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_gate_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.gate_hi(idx(word_index, case.cmd.num_gate_words, "gate_word_hi")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_input_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.input_lo(idx(word_index, case.cmd.num_input_words, "input_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_input_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.input_hi(idx(word_index, case.cmd.num_input_words, "input_word_hi")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.dst_lo(idx(word_index, case.cmd.num_dst_words, "dst_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8mul_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.dst_hi(idx(word_index, case.cmd.num_dst_words, "dst_word_hi")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(int8mul_case_load(0, 8), 0);
        let mut cmd = Int8MulCmd {
            bid: 0,
            iter: 0,
            gate_bank: 0,
            input_bank: 0,
            output_bank: 0,
            op1_col: 0,
            op2_col: 0,
            wr_col: 0,
            gate_row: 0,
            rob_id: 0,
            rs1_lo: 0,
            rs1_hi: 0,
            rs2_lo: 0,
            rs2_hi: 0,
            num_gate_words: 0,
            num_input_words: 0,
            num_dst_words: 0,
        };
        int8mul_case_cmd(&mut cmd as *mut Int8MulCmd);
        assert_eq!(cmd.bid, 8);
        assert_eq!(cmd.iter, 1);
        assert_eq!(cmd.gate_row, 1);
        assert_eq!(cmd.num_gate_words, 2);
        assert_eq!(cmd.num_input_words, 1);
        assert_eq!(cmd.num_dst_words, 1);
        let _ = int8mul_case_gate_word_lo(0);
        let _ = int8mul_case_input_word_lo(0);
        let _ = int8mul_case_dst_word_lo(0);
        assert_eq!(casegen::NUM_CASES, 2);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
