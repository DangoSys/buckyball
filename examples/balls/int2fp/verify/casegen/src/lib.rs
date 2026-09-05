mod casegen;
#[path = "../../../emu/src/model.rs"]
mod model;

use std::cell::RefCell;

use casegen::{Int2FpCase, Int2FpCmd, MAX_DST, MAX_SCALE, MAX_SRC};

thread_local! {
    static CURRENT: RefCell<Option<Int2FpCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn int2fp_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&Int2FpCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("int2fp_case: no case loaded; call int2fp_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn int2fp_case_cmd(out_ptr: *mut Int2FpCmd) {
    if out_ptr.is_null() {
        panic!("int2fp_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn src_index(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("int2fp_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn int2fp_case_src_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_src_words, "src_word_lo");
        if i >= MAX_SRC {
            panic!("int2fp_case_src_word_lo: word_index out of range");
        }
        case.src_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn int2fp_case_src_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_src_words, "src_word_hi");
        if i >= MAX_SRC {
            panic!("int2fp_case_src_word_hi: word_index out of range");
        }
        case.src_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn int2fp_case_scale_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_scale_words, "scale_word_lo");
        if i >= MAX_SCALE {
            panic!("int2fp_case_scale_word_lo: word_index out of range");
        }
        case.scale_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn int2fp_case_scale_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_scale_words, "scale_word_hi");
        if i >= MAX_SCALE {
            panic!("int2fp_case_scale_word_hi: word_index out of range");
        }
        case.scale_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn int2fp_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_dst_words, "dst_word_lo");
        if i >= MAX_DST {
            panic!("int2fp_case_dst_word_lo: word_index out of range");
        }
        case.dst_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn int2fp_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = src_index(word_index, case.cmd.num_dst_words, "dst_word_hi");
        if i >= MAX_DST {
            panic!("int2fp_case_dst_word_hi: word_index out of range");
        }
        case.dst_hi(i)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(int2fp_case_load(0, 4), 0);
        let mut cmd = Int2FpCmd {
            bid: 0,
            iter: 0,
            relu: 0,
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
            num_scale_words: 0,
            num_dst_words: 0,
        };
        int2fp_case_cmd(&mut cmd as *mut Int2FpCmd);
        assert_eq!(cmd.bid, 4);
        assert_eq!(cmd.iter, 4);
        assert_eq!(cmd.relu, 0);
        assert_eq!(cmd.op1_col, 1);
        assert_eq!(cmd.op2_col, 1);
        assert_eq!(cmd.wr_col, 1);
        assert_eq!(cmd.num_src_words, 4);
        assert_eq!(cmd.num_scale_words, 4);
        assert_eq!(cmd.num_dst_words, 4);
        let _lo = int2fp_case_src_word_lo(0);
        let _hi = int2fp_case_src_word_hi(0);
        let _slo = int2fp_case_scale_word_lo(0);
        let _dlo = int2fp_case_dst_word_lo(0);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
