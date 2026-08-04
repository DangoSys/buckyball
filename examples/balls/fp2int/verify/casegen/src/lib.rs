mod casegen;
#[path = "../../../emu/src/model.rs"]
mod model;

use std::cell::RefCell;

use casegen::{Fp2IntCase, Fp2IntCmd, MAX_WORDS};

thread_local! {
  static CURRENT: RefCell<Option<Fp2IntCase>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn fp2int_ref_i32(fp_bits: u32, scale_bits: u32) -> i32 {
  model::fp2int_i32_bits(fp_bits, scale_bits)
}

#[no_mangle]
pub extern "C" fn fp2int_ref_i8(fp_bits: u32, scale_bits: u32) -> i32 {
  i32::from(model::fp2int_i8_bits(fp_bits, scale_bits))
}

#[no_mangle]
pub extern "C" fn fp2int_case_load(seed: u32, index: u32, bid: u32) -> i32 {
  let case = casegen::gen_case(seed, index, bid);
  let nsrc = case.cmd.num_src_words as usize;
  if nsrc > MAX_WORDS {
    panic!("fp2int_case_load: num_src_words out of range {nsrc}");
  }
  CURRENT.with(|c| *c.borrow_mut() = Some(case));
  0
}

fn current_case<F: FnOnce(&Fp2IntCase) -> R, R>(f: F) -> R {
  CURRENT.with(|c| match c.borrow().as_ref() {
    Some(case) => f(case),
    None => panic!("fp2int_case: no case loaded; call fp2int_case_load first"),
  })
}

#[no_mangle]
pub extern "C" fn fp2int_case_cmd(out_ptr: *mut Fp2IntCmd) {
  current_case(|case| unsafe {
    if out_ptr.is_null() {
      panic!("fp2int_case_cmd: null out_ptr");
    }
    *out_ptr = case.cmd;
  });
}

#[no_mangle]
pub extern "C" fn fp2int_case_src_word_lo(word_index: u32) -> u64 {
  if word_index as usize >= MAX_WORDS {
    panic!("fp2int_case_src_word_lo: word_index out of range");
  }
  current_case(|case| case.word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn fp2int_case_src_word_hi(word_index: u32) -> u64 {
  if word_index as usize >= MAX_WORDS {
    panic!("fp2int_case_src_word_hi: word_index out of range");
  }
  current_case(|case| case.word_hi(word_index as usize))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dpi_load_then_cmd_and_words() {
    assert_eq!(fp2int_case_load(0x1234, 0, 3), 0);
    let mut cmd = Fp2IntCmd {
      bid: 0,
      funct7: 0,
      iter: 0,
      scale_bits: 0,
      op1_bank: 0,
      wr_bank: 0,
      op1_col: 0,
      wr_col: 0,
      rob_id: 0,
      num_src_words: 0,
    };
    fp2int_case_cmd(&mut cmd as *mut Fp2IntCmd);
    assert_eq!(cmd.bid, 3);
    assert_eq!(cmd.funct7, 51);
    assert_eq!(cmd.iter, 4);
    assert_eq!(cmd.op1_col, 1);
    assert_eq!(cmd.num_src_words, 4);
    let _lo = fp2int_case_src_word_lo(0);
    let _hi = fp2int_case_src_word_hi(0);
  }

  #[test]
  #[should_panic(expected = "no case loaded")]
  fn current_case_panics_without_load() {
    CURRENT.with(|c| *c.borrow_mut() = None);
    current_case(|_| ());
  }
}
