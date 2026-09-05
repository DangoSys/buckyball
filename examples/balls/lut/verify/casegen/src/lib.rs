mod casegen;

use std::cell::RefCell;

use casegen::{LutCase, LutCmd};

thread_local! {
    static CURRENT: RefCell<Option<LutCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn lut_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&LutCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("lut_case: no case loaded; call lut_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn lut_case_cmd(out_ptr: *mut LutCmd) {
    if out_ptr.is_null() {
        panic!("lut_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn src_index(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("lut_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn lut_case_src_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.src_lo(src_index(word_index, case.cmd.num_src_words, "src_word_lo")))
}

#[no_mangle]
pub extern "C" fn lut_case_src_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.src_hi(src_index(word_index, case.cmd.num_src_words, "src_word_hi")))
}

#[no_mangle]
pub extern "C" fn lut_case_lut_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.lut_lo(src_index(word_index, case.cmd.num_lut_words, "lut_word_lo")))
}

#[no_mangle]
pub extern "C" fn lut_case_lut_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.lut_hi(src_index(word_index, case.cmd.num_lut_words, "lut_word_hi")))
}

#[no_mangle]
pub extern "C" fn lut_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| case.dst_lo(src_index(word_index, case.cmd.num_dst_words, "dst_word_lo")))
}

#[no_mangle]
pub extern "C" fn lut_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| case.dst_hi(src_index(word_index, case.cmd.num_dst_words, "dst_word_hi")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(lut_case_load(0, 5), 0);
        let mut cmd = LutCmd {
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
            num_src_words: 0,
            num_lut_words: 0,
            num_dst_words: 0,
        };
        lut_case_cmd(&mut cmd as *mut LutCmd);
        assert_eq!(cmd.bid, 5);
        assert_eq!(cmd.iter, 4);
        assert_eq!(cmd.num_src_words, 4);
        assert_eq!(cmd.num_lut_words, 16);
        assert_eq!(cmd.num_dst_words, 4);
        let _ = lut_case_src_word_lo(0);
        let _ = lut_case_lut_word_lo(0);
        let _ = lut_case_dst_word_lo(0);
        assert_eq!(casegen::NUM_CASES, 4);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
