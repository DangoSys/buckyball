mod casegen;
#[path = "../../../emu/src/model.rs"]
mod model;

use std::cell::RefCell;

use casegen::{ToInt8Case, ToInt8Cmd, MAX_WORDS};

thread_local! {
  static CURRENT: RefCell<Option<ToInt8Case>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn toint8_ref_i8(fp_bits: u32, scale_bits: u32) -> i32 {
    i32::from(model::toint8_i8_bits(fp_bits, scale_bits))
}

#[no_mangle]
pub extern "C" fn toint8_quant_scale_bits(da_bits: u32) -> u32 {
    model::fp32_divide(1.0f32.to_bits(), da_bits)
}

#[no_mangle]
pub extern "C" fn toint8_case_load(index: u32, bid: u32) -> i32 {
    let case = casegen::gen_case(index, bid);
    let nsrc = case.cmd.num_src_words as usize;
    if nsrc > MAX_WORDS {
        panic!("toint8_case_load: num_src_words out of range {nsrc}");
    }
    CURRENT.with(|c| *c.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&ToInt8Case) -> R, R>(f: F) -> R {
    CURRENT.with(|c| match c.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("toint8_case: no case loaded; call toint8_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn toint8_case_cmd(out_ptr: *mut ToInt8Cmd) {
    current_case(|case| unsafe {
        if out_ptr.is_null() {
            panic!("toint8_case_cmd: null out_ptr");
        }
        let mut cmd = case.cmd;
        cmd.da_bits = model::toint8_da_bits(&case.input_words);
        *out_ptr = cmd;
    });
}

#[no_mangle]
pub extern "C" fn toint8_case_src_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("toint8_case_src_word_lo: word_index out of range");
    }
    current_case(|case| case.word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn toint8_case_src_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("toint8_case_src_word_hi: word_index out of range");
    }
    current_case(|case| case.word_hi(word_index as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(toint8_case_load(0, 3), 0);
        let mut cmd = ToInt8Cmd {
            bid: 0,
            iter: 0,
            da_bits: 0,
            op1_bank: 0,
            wr_bank: 0,
            op1_col: 0,
            wr_col: 0,
            rob_id: 0,
            num_src_words: 0,
        };
        toint8_case_cmd(&mut cmd as *mut ToInt8Cmd);
        assert_eq!(cmd.bid, 3);
        assert_eq!(cmd.iter, 1);
        assert_eq!(cmd.op1_col, 4);
        assert_eq!(cmd.num_src_words, 4);
        let _lo = toint8_case_src_word_lo(0);
        let _hi = toint8_case_src_word_hi(0);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
