mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{MaxPoolCase, MaxPoolCmd, MAX_DST, MAX_INPUT_WORDS};

thread_local! {
    static CURRENT: RefCell<Option<MaxPoolCase>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn maxpool_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid);
    CURRENT.with(|current| *current.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&MaxPoolCase) -> R, R>(f: F) -> R {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("maxpool_case: no case loaded; call maxpool_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn maxpool_case_cmd(out_ptr: *mut MaxPoolCmd) {
    if out_ptr.is_null() {
        panic!("maxpool_case_cmd: null out_ptr");
    }
    current_case(|case| unsafe {
        *out_ptr = case.cmd;
    });
}

fn idx(index: u32, limit: u32, what: &str) -> usize {
    if index >= limit {
        panic!("maxpool_case_{what}: index {index} out of range {limit}");
    }
    index as usize
}

#[no_mangle]
pub extern "C" fn maxpool_case_input_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = idx(word_index, case.cmd.num_input_words, "input_word_lo");
        if i >= MAX_INPUT_WORDS {
            panic!("maxpool_case_input_word_lo: word_index out of range");
        }
        case.input_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn maxpool_case_input_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = idx(word_index, case.cmd.num_input_words, "input_word_hi");
        if i >= MAX_INPUT_WORDS {
            panic!("maxpool_case_input_word_hi: word_index out of range");
        }
        case.input_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn maxpool_case_input_addr(word_index: u32) -> u32 {
    current_case(|case| {
        let i = idx(word_index, case.cmd.num_input_words, "input_addr");
        if i >= MAX_INPUT_WORDS {
            panic!("maxpool_case_input_addr: word_index out of range");
        }
        case.input_addr[i]
    })
}

#[no_mangle]
pub extern "C" fn maxpool_case_dst_word_lo(word_index: u32) -> u64 {
    current_case(|case| {
        let i = idx(word_index, case.cmd.num_dst_words, "dst_word_lo");
        if i >= MAX_DST {
            panic!("maxpool_case_dst_word_lo: word_index out of range");
        }
        case.dst_lo(i)
    })
}

#[no_mangle]
pub extern "C" fn maxpool_case_dst_word_hi(word_index: u32) -> u64 {
    current_case(|case| {
        let i = idx(word_index, case.cmd.num_dst_words, "dst_word_hi");
        if i >= MAX_DST {
            panic!("maxpool_case_dst_word_hi: word_index out of range");
        }
        case.dst_hi(i)
    })
}

#[no_mangle]
pub extern "C" fn maxpool_case_dst_addr(word_index: u32) -> u32 {
    current_case(|case| {
        let i = idx(word_index, case.cmd.num_dst_words, "dst_addr");
        if i >= MAX_DST {
            panic!("maxpool_case_dst_addr: word_index out of range");
        }
        case.dst_addr[i]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(maxpool_case_load(0, 6), 0);
        let mut cmd = MaxPoolCmd {
            bid: 0,
            iter: 0,
            op1_bank: 0,
            wr_bank: 0,
            op1_col: 0,
            wr_col: 0,
            rob_id: 0,
            rs1_lo: 0,
            rs1_hi: 0,
            rs2_lo: 0,
            rs2_hi: 0,
            input_base: 0,
            output_base: 0,
            output_stride: 0,
            input_side: 0,
            output_side: 0,
            kernel: 0,
            stride: 0,
            padding: 0,
            start_row: 0,
            start_col: 0,
            num_input_words: 0,
            num_dst_words: 0,
        };
        maxpool_case_cmd(&mut cmd as *mut MaxPoolCmd);
        assert_eq!(cmd.bid, 6);
        assert_eq!(cmd.iter, 9);
        assert_eq!(cmd.num_input_words, 36);
        assert_eq!(cmd.num_dst_words, 9);
        let _ = maxpool_case_input_word_lo(0);
        let _ = maxpool_case_input_word_hi(0);
        let _ = maxpool_case_input_addr(0);
        let _ = maxpool_case_dst_word_lo(0);
        let _ = maxpool_case_dst_addr(0);
        assert_eq!(casegen::NUM_CASES, 9);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
