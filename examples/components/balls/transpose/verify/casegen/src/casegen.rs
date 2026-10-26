use crate::model;

pub const BANK_ROW_BYTES: usize = 16;
pub const MAX_ITER: usize = 16;
pub const MAX_COLS: usize = 2;
pub const MAX_WORDS: usize = MAX_ITER * MAX_COLS;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransposeCmd {
    pub iter: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
    pub op1_group: u32,
    pub wr_group: u32,
    pub op1_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub elem_bits: u32,
    pub num_src_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransposeCase {
    pub cmd: TransposeCmd,
    pub src: Vec<u8>,
    pub dst: Vec<u8>,
}

impl TransposeCase {
    pub fn src_word_lo(&self, word_index: usize) -> u64 {
        word_lo(&self.src, word_index)
    }
    pub fn src_word_hi(&self, word_index: usize) -> u64 {
        word_hi(&self.src, word_index)
    }
    pub fn dst_word_lo(&self, word_index: usize) -> u64 {
        word_lo(&self.dst, word_index)
    }
    pub fn dst_word_hi(&self, word_index: usize) -> u64 {
        word_hi(&self.dst, word_index)
    }
}

fn word_lo(data: &[u8], word_index: usize) -> u64 {
    let off = word_index * BANK_ROW_BYTES;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(buf)
}

fn word_hi(data: &[u8], word_index: usize) -> u64 {
    let off = word_index * BANK_ROW_BYTES + 8;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(buf)
}

// Word `i` of a case is bank row `i / cols`, group `i % cols`: each row spans `cols` groups.
pub fn gen_case(seed: u32, index: u32) -> TransposeCase {
    match index {
        0 => directed_i8(),
        1 => directed_i32(),
        2 => directed_i8_two_groups(),
        _ => random_case(seed, index),
    }
}

fn directed_i8() -> TransposeCase {
    let iter = 16usize;
    let elem_bytes = 1usize;
    let cols = 1usize;
    let w = cols * (BANK_ROW_BYTES / elem_bytes);
    let total = iter * w;
    let src: Vec<u8> = (0..total).map(|i| i as u8).collect();
    let dst = model::transpose_bytes(&src, iter, w, elem_bytes);
    let nwords = model::num_words(total * elem_bytes) as u32;
    TransposeCase {
        cmd: TransposeCmd {
            iter: iter as u32,
            op1_bank: 0,
            wr_bank: 1,
            op1_group: 0,
            wr_group: 0,
            op1_col: cols as u32,
            wr_col: cols as u32,
            rob_id: 3,
            elem_bits: (elem_bytes * 8) as u32,
            num_src_words: nwords,
            num_dst_words: nwords,
        },
        src,
        dst,
    }
}

fn directed_i32() -> TransposeCase {
    let iter = 8usize;
    let elem_bytes = 4usize;
    let cols = 1usize;
    let w = cols * (BANK_ROW_BYTES / elem_bytes);
    let total = iter * w;
    let src: Vec<u8> = (0..total)
        .map(|i| i as u32)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let dst = model::transpose_bytes(&src, iter, w, elem_bytes);
    let nwords = model::num_words(total * elem_bytes) as u32;
    TransposeCase {
        cmd: TransposeCmd {
            iter: iter as u32,
            op1_bank: 0,
            wr_bank: 1,
            op1_group: 0,
            wr_group: 0,
            op1_col: cols as u32,
            wr_col: cols as u32,
            rob_id: 5,
            elem_bits: (elem_bytes * 8) as u32,
            num_src_words: nwords,
            num_dst_words: nwords,
        },
        src,
        dst,
    }
}

// Mirrors transpose_i8_16x32_test.c: a 16x32 i8 matrix whose rows span two groups.
fn directed_i8_two_groups() -> TransposeCase {
    let iter = 16usize;
    let elem_bytes = 1usize;
    let cols = 2usize;
    let w = cols * (BANK_ROW_BYTES / elem_bytes);
    let total = iter * w;
    let src: Vec<u8> = (0..iter)
        .flat_map(|r| (0..w).map(move |c| (r + c + 1) as u8))
        .collect();
    let dst = model::transpose_bytes(&src, iter, w, elem_bytes);
    let nwords = model::num_words(total * elem_bytes) as u32;
    TransposeCase {
        cmd: TransposeCmd {
            iter: iter as u32,
            op1_bank: 0,
            wr_bank: 1,
            op1_group: 2,
            wr_group: 1,
            op1_col: cols as u32,
            wr_col: cols as u32,
            rob_id: 6,
            elem_bits: (elem_bytes * 8) as u32,
            num_src_words: nwords,
            num_dst_words: nwords,
        },
        src,
        dst,
    }
}

fn random_case(seed: u32, index: u32) -> TransposeCase {
    let mut rng = Rng::new(seed, index);
    let elem_bits = if rng.next() & 1 == 0 { 8u32 } else { 32u32 };
    let elem_bytes = (elem_bits / 8) as usize;
    let iter_pool = [1u32, 2, 4, 8, 16];
    let iter = iter_pool[(rng.next() as usize) % iter_pool.len()] as usize;
    let cols = (rng.next() as usize) % MAX_COLS + 1;
    let w = cols * (BANK_ROW_BYTES / elem_bytes);

    // Keep banks in 0..7 so the same cases work on configs with bankNum=8 or 16.
    let op1_bank = rng.range(0, 7);
    let mut wr_bank = rng.range(0, 7);
    if wr_bank == op1_bank {
        wr_bank = (wr_bank + 1) & 7;
    }

    let op1_group = rng.range(0, 3);
    let wr_group = rng.range(0, 3);

    let total = iter * w;
    let mut src = vec![0u8; total * elem_bytes];
    for b in src.iter_mut() {
        *b = (rng.next() & 0xFF) as u8;
    }
    let dst = model::transpose_bytes(&src, iter, w, elem_bytes);
    let nwords = model::num_words(total * elem_bytes) as u32;

    TransposeCase {
        cmd: TransposeCmd {
            iter: iter as u32,
            op1_bank,
            wr_bank,
            op1_group,
            wr_group,
            op1_col: cols as u32,
            wr_col: cols as u32,
            rob_id: rng.range(0, 15),
            elem_bits,
            num_src_words: nwords,
            num_dst_words: nwords,
        },
        src,
        dst,
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
