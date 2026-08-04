mod casegen;
#[path = "../../../emu/src/model.rs"]
mod model;

use std::cell::RefCell;

use casegen::{Int2FpCase, Int2FpCmd, MAX_WORDS};

thread_local! {
  static CURRENT: RefCell<Option<Int2FpCase>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn int2fp_ref_fp32(value: i32, scale_bits: u32) -> u32 {
  model::int2fp_fp32_bits(value, scale_bits)
}

#[no_mangle]
pub extern "C" fn int2fp_ref_i8(value: i32, scale_bits: u32) -> i32 {
  i32::from(model::int2fp_i8_bits(value, scale_bits))
}

#[no_mangle]
pub extern "C" fn int2fp_case_load(seed: u32, index: u32, bid: u32) -> i32 {
  let case = casegen::gen_case(seed, index, bid);
  let nsrc = case.cmd.num_src_words as usize;
  if nsrc == 0 || nsrc > MAX_WORDS {
    panic!("int2fp_case_load: num_src_words out of range {nsrc}");
  }
  CURRENT.with(|c| *c.borrow_mut() = Some(case));
  0
}

fn current_case<F: FnOnce(&Int2FpCase) -> R, R>(f: F) -> R {
  CURRENT.with(|c| match c.borrow().as_ref() {
    Some(case) => f(case),
    None => panic!("int2fp_case: no case loaded; call int2fp_case_load first"),
  })
}

#[no_mangle]
pub extern "C" fn int2fp_case_cmd(out_ptr: *mut Int2FpCmd) {
  current_case(|case| unsafe {
    if out_ptr.is_null() {
      panic!("int2fp_case_cmd: null out_ptr");
    }
    *out_ptr = case.cmd;
  });
}

#[no_mangle]
pub extern "C" fn int2fp_case_src_word_lo(word_index: u32) -> u64 {
  if word_index as usize >= MAX_WORDS {
    panic!("int2fp_case_src_word_lo: word_index out of range");
  }
  current_case(|case| case.word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn int2fp_case_src_word_hi(word_index: u32) -> u64 {
  if word_index as usize >= MAX_WORDS {
    panic!("int2fp_case_src_word_hi: word_index out of range");
  }
  current_case(|case| case.word_hi(word_index as usize))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dpi_load_then_cmd_and_words() {
    assert_eq!(int2fp_case_load(0x1234, 0, 4), 0);
    let mut cmd = Int2FpCmd {
      bid: 0,
      funct7: 0,
      iter: 0,
      scale_bits: 0,
      output_mode: 0,
      op1_bank: 0,
      wr_bank: 0,
      op1_col: 0,
      wr_col: 0,
      rob_id: 0,
      num_src_words: 0,
    };
    int2fp_case_cmd(&mut cmd as *mut Int2FpCmd);
    assert_eq!(cmd.bid, 4);
    assert_eq!(cmd.funct7, 52);
    assert_eq!(cmd.iter, 16);
    assert_eq!(cmd.output_mode, 0);
    assert_eq!(cmd.num_src_words, 16);
    let _lo = int2fp_case_src_word_lo(0);
    let _hi = int2fp_case_src_word_hi(0);
  }

  #[test]
  #[should_panic(expected = "no case loaded")]
  fn current_case_panics_without_load() {
    CURRENT.with(|c| *c.borrow_mut() = None);
    current_case(|_| ());
  }
}
