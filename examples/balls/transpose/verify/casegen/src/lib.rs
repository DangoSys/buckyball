mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{TransposeCase, TransposeCmd, MAX_WORDS};

thread_local! {
    static CURRENT: RefCell<Option<TransposeCase>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn transpose_case_load(seed: u32, index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(seed, index, bid);
    let nsrc = case.cmd.num_src_words as usize;
    let ndst = case.cmd.num_dst_words as usize;
    if nsrc > MAX_WORDS || ndst > MAX_WORDS {
        panic!("transpose_case_load: word count out of range src={nsrc} dst={ndst}");
    }
    CURRENT.with(|c| *c.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&TransposeCase) -> R, R>(f: F) -> R {
    CURRENT.with(|c| match c.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("transpose_case: no case loaded; call transpose_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn transpose_case_cmd(out_ptr: *mut TransposeCmd) {
    current_case(|case| unsafe {
        if out_ptr.is_null() {
            panic!("transpose_case_cmd: null out_ptr");
        }
        *out_ptr = case.cmd;
    });
}

#[no_mangle]
pub extern "C" fn transpose_case_src_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_src_word_lo: word_index out of range");
    }
    current_case(|case| case.src_word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn transpose_case_src_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_src_word_hi: word_index out of range");
    }
    current_case(|case| case.src_word_hi(word_index as usize))
}

#[no_mangle]
pub extern "C" fn transpose_case_dst_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_dst_word_lo: word_index out of range");
    }
    current_case(|case| case.dst_word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn transpose_case_dst_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("transpose_case_dst_word_hi: word_index out of range");
    }
    current_case(|case| case.dst_word_hi(word_index as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(transpose_case_load(0x1234, 0, 0), 0);
        let mut cmd = TransposeCmd {
            bid: 0,
            iter: 0,
            op1_bank: 0,
            wr_bank: 0,
            op1_col: 0,
            wr_col: 0,
            rob_id: 0,
            elem_bits: 0,
            num_src_words: 0,
            num_dst_words: 0,
        };
        transpose_case_cmd(&mut cmd as *mut TransposeCmd);
        assert_eq!(cmd.iter, 16);
        assert_eq!(cmd.elem_bits, 8);
        assert_eq!(cmd.num_src_words, 16);
        let _lo = transpose_case_src_word_lo(0);
        let _hi = transpose_case_src_word_hi(0);
        let _dlo = transpose_case_dst_word_lo(0);
        let _dhi = transpose_case_dst_word_hi(0);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
