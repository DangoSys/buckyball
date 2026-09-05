mod casegen;

use std::cell::RefCell;

use casegen::{Int8AddCase, Int8AddCmd};

thread_local! {
    static CURRENT: RefCell<Option<Int8AddCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn int8add_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&Int8AddCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("int8add_case: no case loaded; call int8add_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn int8add_case_cmd(out_ptr: *mut Int8AddCmd) {
    if out_ptr.is_null() {
        panic!("int8add_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn idx(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("int8add_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn int8add_case_lhs_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.lhs_lo(idx(word_index, case.cmd.num_lhs_words, "lhs_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8add_case_lhs_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.lhs_hi(idx(word_index, case.cmd.num_lhs_words, "lhs_word_hi")))
}

#[no_mangle]
pub extern "C" fn int8add_case_rhs_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.rhs_lo(idx(word_index, case.cmd.num_rhs_words, "rhs_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8add_case_rhs_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.rhs_hi(idx(word_index, case.cmd.num_rhs_words, "rhs_word_hi")))
}

#[no_mangle]
pub extern "C" fn int8add_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.dst_lo(idx(word_index, case.cmd.num_dst_words, "dst_word_lo")))
}

#[no_mangle]
pub extern "C" fn int8add_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.dst_hi(idx(word_index, case.cmd.num_dst_words, "dst_word_hi")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(int8add_case_load(0, 7), 0);
        let mut cmd = Int8AddCmd {
            relu: 0,
            bid: 0,
            iter: 0,
            op1_bank: 0,
            op2_bank: 0,
            wr_bank: 0,
            op1_col: 0,
            op2_col: 0,
            wr_col: 0,
            rob_id: 0,
            rs1_lo: 0,
            rs1_hi: 0,
            rs2_lo: 0,
            rs2_hi: 0,
            num_lhs_words: 0,
            num_rhs_words: 0,
            num_dst_words: 0,
        };
        int8add_case_cmd(&mut cmd as *mut Int8AddCmd);
        assert_eq!(cmd.bid, 7);
        assert_eq!(cmd.iter, 7);
        assert_eq!(cmd.relu, 0);
        assert_eq!(cmd.num_lhs_words, 7);
        assert_eq!(cmd.num_rhs_words, 7);
        assert_eq!(cmd.num_dst_words, 7);
        let _ = int8add_case_lhs_word_lo(0);
        let _ = int8add_case_rhs_word_lo(0);
        let _ = int8add_case_dst_word_lo(0);
        assert_eq!(casegen::NUM_CASES, 2);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
