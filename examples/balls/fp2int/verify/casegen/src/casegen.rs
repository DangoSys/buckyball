pub const WORDS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp2IntCase {
  pub bid: u32,
  pub funct7: u32,
  pub iter: u32,
  pub scale_bits: u32,
  pub op1_bank: u32,
  pub op2_bank: u32,
  pub wr_bank: u32,
  pub op1_col: u32,
  pub op2_col: u32,
  pub wr_col: u32,
  pub meta_bank: u32,
  pub rob_id: u32,
  pub is_sub: bool,
  pub sub_rob_id: u32,
  pub input_words: [u128; WORDS],
}

impl Fp2IntCase {
  pub fn word_lo(&self, index: usize) -> u64 {
    self.input_words[index] as u64
  }

  pub fn word_hi(&self, index: usize) -> u64 {
    (self.input_words[index] >> 64) as u64
  }
}

pub fn gen_case(seed: u32, index: u32) -> Fp2IntCase {
  if index == 0 {
    return directed_i32_case();
  }

  random_i32_case(seed, index)
}

fn directed_i32_case() -> Fp2IntCase {
  Fp2IntCase {
    bid: 5,
    funct7: 51,
    iter: WORDS as u32,
    scale_bits: 0x3F80_0000,
    op1_bank: 0,
    op2_bank: 0,
    wr_bank: 1,
    op1_col: 1,
    op2_col: 0,
    wr_col: 1,
    meta_bank: 0,
    rob_id: 3,
    is_sub: false,
    sub_rob_id: 0,
    input_words: [
      0xBF80_0000_4040_0000_4000_0000_3F80_0000,
      0x40A0_0000_4080_0000_0000_0000_C000_0000,
      0x42C8_0000_3F00_0000_C120_0000_4120_0000,
      0xC100_0000_4100_0000_40E0_0000_C2C8_0000,
    ],
  }
}

fn random_i32_case(seed: u32, index: u32) -> Fp2IntCase {
  let mut rng = Rng::new(seed, index);
  let op1_bank = rng.range(0, 31);
  let mut wr_bank = rng.range(0, 31);
  if wr_bank == op1_bank {
    wr_bank = (wr_bank + 1) & 31;
  }

  Fp2IntCase {
    bid: 5,
    funct7: 51,
    iter: WORDS as u32,
    scale_bits: scale_pool(rng.range(0, 3)),
    op1_bank,
    op2_bank: 0,
    wr_bank,
    op1_col: 1,
    op2_col: 0,
    wr_col: 1,
    meta_bank: 0,
    rob_id: rng.range(0, 15),
    is_sub: false,
    sub_rob_id: 0,
    input_words: [
      random_word(&mut rng),
      random_word(&mut rng),
      random_word(&mut rng),
      random_word(&mut rng),
    ],
  }
}

fn random_word(rng: &mut Rng) -> u128 {
  let mut word = 0u128;
  for lane in 0..4 {
    word |= u128::from(fp_pool(rng.range(0, 15))) << (lane * 32);
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

fn fp_pool(index: u32) -> u32 {
  match index {
    0 => 0x0000_0000,
    1 => 0x8000_0000,
    2 => 0x3F80_0000,
    3 => 0xBF80_0000,
    4 => 0x4000_0000,
    5 => 0xC000_0000,
    6 => 0x3F00_0000,
    7 => 0xBF00_0000,
    8 => 0x3FC0_0000,
    9 => 0xBFC0_0000,
    10 => 0x4120_0000,
    11 => 0xC120_0000,
    12 => 0x42C8_0000,
    13 => 0xC2C8_0000,
    14 => 0x4F00_0000,
    15 => 0xCF00_0000,
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

  #[test]
  fn case_zero_is_smoke_case() {
    let case = gen_case(0x1234, 0);

    assert_eq!(case.bid, 5);
    assert_eq!(case.funct7, 51);
    assert_eq!(case.iter, WORDS as u32);
    assert_eq!(case.scale_bits, 0x3F80_0000);
    assert_eq!(case.op1_bank, 0);
    assert_eq!(case.wr_bank, 1);
    assert_eq!(case.op1_col, 1);
    assert_eq!(case.wr_col, 1);
    assert_eq!(case.rob_id, 3);
    assert_eq!(case.input_words[0], 0xBF80_0000_4040_0000_4000_0000_3F80_0000);
  }

  #[test]
  fn random_cases_are_deterministic_and_legal() {
    let a = gen_case(0xCAFE_BABE, 7);
    let b = gen_case(0xCAFE_BABE, 7);

    assert_eq!(a, b);
    assert_eq!(a.bid, 5);
    assert_eq!(a.funct7, 51);
    assert_eq!(a.iter, WORDS as u32);
    assert_ne!(a.op1_bank, a.wr_bank);
    assert!(a.op1_bank < 32);
    assert!(a.wr_bank < 32);
    assert!(a.rob_id < 16);
    assert_eq!(a.op1_col, 1);
    assert_eq!(a.wr_col, 1);
  }
}
