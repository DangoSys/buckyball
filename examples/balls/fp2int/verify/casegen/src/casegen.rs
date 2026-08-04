pub const FUNCT7: u32 = 51;
pub const WORDS: usize = 4;
pub const GROUPS: usize = 4;
pub const MAX_WORDS: usize = WORDS * GROUPS;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp2IntCmd {
  pub bid: u32,
  pub funct7: u32,
  pub iter: u32,
  pub scale_bits: u32,
  pub op1_bank: u32,
  pub wr_bank: u32,
  pub op1_col: u32,
  pub wr_col: u32,
  pub rob_id: u32,
  pub num_src_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp2IntCase {
  pub cmd: Fp2IntCmd,
  pub input_words: [u128; MAX_WORDS],
}

impl Fp2IntCase {
  pub fn word_lo(&self, index: usize) -> u64 {
    self.input_words[index] as u64
  }

  pub fn word_hi(&self, index: usize) -> u64 {
    (self.input_words[index] >> 64) as u64
  }

  #[allow(dead_code)]
  pub fn is_i8(&self) -> bool {
    self.cmd.op1_col == 4 && self.cmd.wr_col == 1
  }
}

pub fn gen_case(seed: u32, index: u32, bid: u32) -> Fp2IntCase {
  match index {
    0 => directed_i32_case(bid),
    1 => directed_i8_case(bid),
    _ => random_i32_case(seed, index, bid),
  }
}

fn directed_i32_case(bid: u32) -> Fp2IntCase {
  let mut input_words = [0u128; MAX_WORDS];
  input_words[0] = 0xBF80_0000_4040_0000_4000_0000_3F80_0000;
  input_words[1] = 0x40A0_0000_4080_0000_0000_0000_C000_0000;
  input_words[2] = 0x42C8_0000_3F00_0000_C120_0000_4120_0000;
  input_words[3] = 0xC100_0000_4100_0000_40E0_0000_C2C8_0000;

  Fp2IntCase {
    cmd: Fp2IntCmd {
      bid,
      funct7: FUNCT7,
      iter: WORDS as u32,
      scale_bits: 0x3F80_0000,
      op1_bank: 0,
      wr_bank: 1,
      op1_col: 1,
      wr_col: 1,
      rob_id: 3,
      num_src_words: WORDS as u32,
    },
    input_words,
  }
}

fn directed_i8_case(bid: u32) -> Fp2IntCase {
  let mut input_words = [0u128; MAX_WORDS];
  let vals: [u32; 16] = [
    0x3E00_0000, // 0.125
    0xBE00_0000, // -0.125
    0x3E80_0000, // 0.25
    0xBE80_0000, // -0.25
    0x3F40_0000, // 0.75
    0xBF40_0000, // -0.75
    0x3FA0_0000, // 1.25
    0xBFA0_0000, // -1.25
    0x3FE0_0000, // 1.75
    0xBFE0_0000, // -1.75
    0x427D_0000, // 63.25
    0x427F_0000, // 63.75
    0xC27F_0000, // -63.75
    0xC281_8000, // -64.75
    0x0000_0000,
    0x8000_0000,
  ];

  for group in 0..GROUPS {
    let mut word = 0u128;
    for lane in 0..4 {
      word |= u128::from(vals[group * 4 + lane]) << (lane * 32);
    }
    input_words[group] = word;
  }

  Fp2IntCase {
    cmd: Fp2IntCmd {
      bid,
      funct7: FUNCT7,
      iter: 1,
      scale_bits: 0x4000_0000, // 2.0
      op1_bank: 0,
      wr_bank: 1,
      op1_col: 4,
      wr_col: 1,
      rob_id: 2,
      num_src_words: GROUPS as u32,
    },
    input_words,
  }
}

fn random_i32_case(seed: u32, index: u32, bid: u32) -> Fp2IntCase {
  let mut rng = Rng::new(seed, index);
  let op1_bank = rng.range(0, 7);
  let mut wr_bank = rng.range(0, 7);
  if wr_bank == op1_bank {
    wr_bank = (wr_bank + 1) & 7;
  }

  let mut input_words = [0u128; MAX_WORDS];
  for i in 0..WORDS {
    input_words[i] = random_word(&mut rng);
  }

  Fp2IntCase {
    cmd: Fp2IntCmd {
      bid,
      funct7: FUNCT7,
      iter: WORDS as u32,
      scale_bits: scale_pool(rng.range(0, 3)),
      op1_bank,
      wr_bank,
      op1_col: 1,
      wr_col: 1,
      rob_id: rng.range(0, 15),
      num_src_words: WORDS as u32,
    },
    input_words,
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
    let case = gen_case(0x1234, 0, 3);

    assert_eq!(case.cmd.bid, 3);
    assert_eq!(case.cmd.funct7, FUNCT7);
    assert_eq!(case.cmd.iter, WORDS as u32);
    assert_eq!(case.cmd.scale_bits, 0x3F80_0000);
    assert_eq!(case.cmd.op1_bank, 0);
    assert_eq!(case.cmd.wr_bank, 1);
    assert_eq!(case.cmd.op1_col, 1);
    assert_eq!(case.cmd.wr_col, 1);
    assert_eq!(case.cmd.rob_id, 3);
    assert_eq!(case.cmd.num_src_words, WORDS as u32);
    assert_eq!(case.input_words[0], 0xBF80_0000_4040_0000_4000_0000_3F80_0000);
  }

  #[test]
  fn case_one_is_i8_case() {
    let case = gen_case(0, 1, 5);
    assert!(case.is_i8());
    assert_eq!(case.cmd.bid, 5);
    assert_eq!(case.cmd.iter, 1);
    assert_eq!(case.cmd.scale_bits, 0x4000_0000);
    assert_eq!(case.cmd.num_src_words, GROUPS as u32);
  }

  #[test]
  fn random_cases_are_deterministic_and_legal() {
    let a = gen_case(0xCAFE_BABE, 7, 3);
    let b = gen_case(0xCAFE_BABE, 7, 3);

    assert_eq!(a, b);
    assert_eq!(a.cmd.bid, 3);
    assert_eq!(a.cmd.funct7, FUNCT7);
    assert_eq!(a.cmd.iter, WORDS as u32);
    assert_ne!(a.cmd.op1_bank, a.cmd.wr_bank);
    assert!(a.cmd.op1_bank < 8);
    assert!(a.cmd.wr_bank < 8);
    assert!(a.cmd.rob_id < 16);
    assert_eq!(a.cmd.op1_col, 1);
    assert_eq!(a.cmd.wr_col, 1);
  }

  #[test]
  fn bid_is_required_arg() {
    let case = gen_case(0, 0, 3);
    assert_eq!(case.cmd.bid, 3);
    let case = gen_case(0, 0, 5);
    assert_eq!(case.cmd.bid, 5);
  }
}
