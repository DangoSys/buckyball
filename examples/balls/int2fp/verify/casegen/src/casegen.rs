pub const FUNCT7: u32 = 52;
pub const GROUPS: usize = 4;
pub const MAX_ITER: usize = 16;
pub const MAX_WORDS: usize = MAX_ITER * GROUPS;
pub const OUTPUT_FP32: u32 = 0;
pub const OUTPUT_INT8: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int2FpCmd {
  pub bid: u32,
  pub funct7: u32,
  pub iter: u32,
  pub scale_bits: u32,
  pub output_mode: u32,
  pub op1_bank: u32,
  pub wr_bank: u32,
  pub op1_col: u32,
  pub wr_col: u32,
  pub rob_id: u32,
  pub num_src_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int2FpCase {
  pub cmd: Int2FpCmd,
  pub input_words: [u128; MAX_WORDS],
}

impl Int2FpCase {
  pub fn word_lo(&self, index: usize) -> u64 {
    self.input_words[index] as u64
  }

  pub fn word_hi(&self, index: usize) -> u64 {
    (self.input_words[index] >> 64) as u64
  }

  #[allow(dead_code)]
  pub fn is_i8(&self) -> bool {
    self.cmd.output_mode == OUTPUT_INT8
      && self.cmd.op1_col == 4
      && self.cmd.wr_col == 1
  }
}

pub fn gen_case(seed: u32, index: u32, bid: u32) -> Int2FpCase {
  match index {
    0 => directed_i32_to_fp32(bid),
    1 => directed_i32_to_i8(bid),
    _ => random_case(seed, index, bid),
  }
}

fn pack_i32s(vals: &[i32]) -> u128 {
  if vals.len() != 4 {
    panic!("pack_i32s: need exactly 4 lanes, got {}", vals.len());
  }
  let mut word = 0u128;
  for (lane, v) in vals.iter().enumerate() {
    word |= u128::from(*v as u32) << (lane * 32);
  }
  word
}

fn directed_i32_to_fp32(bid: u32) -> Int2FpCase {
  let vals: [i32; 64] = [
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8, 1, 2, 3, -1, -2, 0, 4, 5, 10,
    -10, 7, 100, -100, 8, 16, -8, 1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8, 1,
    2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
  ];
  let iter = 16usize;
  let mut input_words = [0u128; MAX_WORDS];
  for w in 0..iter {
    input_words[w] = pack_i32s(&vals[w * 4..w * 4 + 4]);
  }

  Int2FpCase {
    cmd: Int2FpCmd {
      bid,
      funct7: FUNCT7,
      iter: iter as u32,
      scale_bits: 0x3F80_0000,
      output_mode: OUTPUT_FP32,
      op1_bank: 0,
      wr_bank: 1,
      op1_col: 1,
      wr_col: 1,
      rob_id: 3,
      num_src_words: iter as u32,
    },
    input_words,
  }
}

fn directed_i32_to_i8(bid: u32) -> Int2FpCase {
  let vals: [i32; 32] = [
    -1000, -257, -255, -5, -3, -1, 0, 1, 3, 5, 127, 253, 255, 257, 1000, 2, -999, -511, -259, -9,
    -7, -3, 2, 4, 6, 9, 125, 251, 254, 258, 511, 999,
  ];
  let iter = 2usize;
  let mut input_words = [0u128; MAX_WORDS];
  for row in 0..iter {
    for group in 0..GROUPS {
      let base = row * 16 + group * 4;
      input_words[row * GROUPS + group] = pack_i32s(&vals[base..base + 4]);
    }
  }

  Int2FpCase {
    cmd: Int2FpCmd {
      bid,
      funct7: FUNCT7,
      iter: iter as u32,
      scale_bits: 0x3F00_0000,
      output_mode: OUTPUT_INT8,
      op1_bank: 0,
      wr_bank: 1,
      op1_col: 4,
      wr_col: 1,
      rob_id: 2,
      num_src_words: (iter * GROUPS) as u32,
    },
    input_words,
  }
}

fn random_case(seed: u32, index: u32, bid: u32) -> Int2FpCase {
  let mut rng = Rng::new(seed, index);
  let to_i8 = (rng.next() & 1) == 1;
  if to_i8 {
    random_i32_to_i8(&mut rng, bid)
  } else {
    random_i32_to_fp32(&mut rng, bid)
  }
}

fn random_i32_to_fp32(rng: &mut Rng, bid: u32) -> Int2FpCase {
  let iter_pool = [1u32, 2, 4, 16];
  let iter = iter_pool[(rng.next() as usize) % iter_pool.len()] as usize;
  let op1_bank = rng.range(0, 7);
  let mut wr_bank = rng.range(0, 7);
  if wr_bank == op1_bank {
    wr_bank = (wr_bank + 1) & 7;
  }

  let mut input_words = [0u128; MAX_WORDS];
  for w in 0..iter {
    input_words[w] = random_i32_word(rng);
  }

  Int2FpCase {
    cmd: Int2FpCmd {
      bid,
      funct7: FUNCT7,
      iter: iter as u32,
      scale_bits: scale_pool(rng.range(0, 3)),
      output_mode: OUTPUT_FP32,
      op1_bank,
      wr_bank,
      op1_col: 1,
      wr_col: 1,
      rob_id: rng.range(0, 15),
      num_src_words: iter as u32,
    },
    input_words,
  }
}

fn random_i32_to_i8(rng: &mut Rng, bid: u32) -> Int2FpCase {
  let iter_pool = [1u32, 2, 4];
  let iter = iter_pool[(rng.next() as usize) % iter_pool.len()] as usize;
  let op1_bank = rng.range(0, 7);
  let mut wr_bank = rng.range(0, 7);
  if wr_bank == op1_bank {
    wr_bank = (wr_bank + 1) & 7;
  }

  let mut input_words = [0u128; MAX_WORDS];
  for row in 0..iter {
    for group in 0..GROUPS {
      input_words[row * GROUPS + group] = random_i32_word(rng);
    }
  }

  Int2FpCase {
    cmd: Int2FpCmd {
      bid,
      funct7: FUNCT7,
      iter: iter as u32,
      scale_bits: scale_pool(rng.range(0, 3)),
      output_mode: OUTPUT_INT8,
      op1_bank,
      wr_bank,
      op1_col: 4,
      wr_col: 1,
      rob_id: rng.range(0, 15),
      num_src_words: (iter * GROUPS) as u32,
    },
    input_words,
  }
}

fn random_i32_word(rng: &mut Rng) -> u128 {
  let mut word = 0u128;
  for lane in 0..4 {
    word |= u128::from(i32_pool(rng.range(0, 15)) as u32) << (lane * 32);
  }
  word
}

fn scale_pool(index: u32) -> u32 {
  match index {
    0 => 0x3F80_0000,
    1 => 0x4000_0000,
    2 => 0x3F00_0000,
    3 => 0xBF80_0000,
    _ => unreachable!(),
  }
}

fn i32_pool(index: u32) -> i32 {
  match index {
    0 => 0,
    1 => 1,
    2 => -1,
    3 => 2,
    4 => -2,
    5 => 127,
    6 => -128,
    7 => 255,
    8 => -255,
    9 => 1000,
    10 => -1000,
    11 => 16,
    12 => -16,
    13 => i32::MAX,
    14 => i32::MIN,
    15 => 42,
    _ => unreachable!(),
  }
}

struct Rng {
  state: u64,
}

impl Rng {
  fn new(seed: u32, index: u32) -> Self {
    let state = (u64::from(seed) << 32) ^ u64::from(index) ^ 0x9E37_79B9_7F4A_7C15;
    Self { state }
  }

  fn next(&mut self) -> u32 {
    self.state ^= self.state >> 12;
    self.state ^= self.state << 25;
    self.state ^= self.state >> 27;
    ((self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
  }

  fn range(&mut self, lo: u32, hi: u32) -> u32 {
    lo + (self.next() % (hi - lo + 1))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::model;

  #[test]
  fn case_zero_is_fp32_smoke() {
    let case = gen_case(0x1234, 0, 4);
    assert_eq!(case.cmd.bid, 4);
    assert_eq!(case.cmd.funct7, FUNCT7);
    assert_eq!(case.cmd.iter, 16);
    assert_eq!(case.cmd.scale_bits, 0x3F80_0000);
    assert_eq!(case.cmd.output_mode, OUTPUT_FP32);
    assert_eq!(case.cmd.op1_col, 1);
    assert_eq!(case.cmd.wr_col, 1);
    assert_eq!(case.cmd.num_src_words, 16);
    assert!(!case.is_i8());
    assert_eq!(model::int2fp_fp32_bits(1, 0x3F80_0000), 0x3F80_0000);
  }

  #[test]
  fn case_one_is_i8_requant() {
    let case = gen_case(0, 1, 6);
    assert!(case.is_i8());
    assert_eq!(case.cmd.bid, 6);
    assert_eq!(case.cmd.iter, 2);
    assert_eq!(case.cmd.scale_bits, 0x3F00_0000);
    assert_eq!(case.cmd.output_mode, OUTPUT_INT8);
    assert_eq!(case.cmd.op1_col, 4);
    assert_eq!(case.cmd.wr_col, 1);
    assert_eq!(case.cmd.num_src_words, 8);
    assert_eq!(model::int2fp_i8_bits(-1000, 0x3F00_0000), -128);
    assert_eq!(model::int2fp_i8_bits(5, 0x3F00_0000), 2);
  }

  #[test]
  fn random_cases_deterministic_and_legal() {
    let a = gen_case(0xCAFE_BABE, 7, 4);
    let b = gen_case(0xCAFE_BABE, 7, 4);
    assert_eq!(a, b);
    assert_eq!(a.cmd.bid, 4);
    assert_eq!(a.cmd.funct7, FUNCT7);
    assert_ne!(a.cmd.op1_bank, a.cmd.wr_bank);
    assert!(a.cmd.op1_bank < 8);
    assert!(a.cmd.wr_bank < 8);
    assert!(a.cmd.rob_id < 16);
    assert!(a.cmd.output_mode == OUTPUT_FP32 || a.cmd.output_mode == OUTPUT_INT8);
    if a.is_i8() {
      assert_eq!(a.cmd.op1_col, 4);
      assert_eq!(a.cmd.wr_col, 1);
      assert_eq!(a.cmd.num_src_words, a.cmd.iter * GROUPS as u32);
    } else {
      assert_eq!(a.cmd.op1_col, 1);
      assert_eq!(a.cmd.wr_col, 1);
      assert_eq!(a.cmd.num_src_words, a.cmd.iter);
    }
  }

  #[test]
  fn bid_is_required_arg() {
    assert_eq!(gen_case(0, 0, 4).cmd.bid, 4);
    assert_eq!(gen_case(0, 0, 6).cmd.bid, 6);
  }
}
