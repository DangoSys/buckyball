use crate::model;

pub const BANK_ROW_BYTES: usize = 16;
pub const MAX_ITER: usize = 16;
pub const MAX_WORDS: usize = MAX_ITER;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransposeCmd {
    pub bid: u32,
    pub iter: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
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

pub fn gen_case(seed: u32, index: u32, bid: u32) -> TransposeCase {
    match index {
        0 => directed_i8(bid),
        1 => directed_i32(bid),
        _ => random_case(seed, index, bid),
    }
}

fn directed_i8(bid: u32) -> TransposeCase {
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
            bid,
            iter: iter as u32,
            op1_bank: 0,
            wr_bank: 1,
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

fn directed_i32(bid: u32) -> TransposeCase {
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
            bid,
            iter: iter as u32,
            op1_bank: 0,
            wr_bank: 1,
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

fn random_case(seed: u32, index: u32, bid: u32) -> TransposeCase {
    let mut rng = Rng::new(seed, index);
    let elem_bits = if rng.next() & 1 == 0 { 8u32 } else { 32u32 };
    let elem_bytes = (elem_bits / 8) as usize;
    let iter_pool = [1u32, 2, 4, 8, 16];
    let iter = iter_pool[(rng.next() as usize) % iter_pool.len()] as usize;
    let cols = 1usize;
    let w = cols * (BANK_ROW_BYTES / elem_bytes);

    // Keep banks in 0..7 so the same cases work on configs with bankNum=8 or 16.
    let op1_bank = rng.range(0, 7);
    let mut wr_bank = rng.range(0, 7);
    if wr_bank == op1_bank {
        wr_bank = (wr_bank + 1) & 7;
    }

    let total = iter * w;
    let mut src = vec![0u8; total * elem_bytes];
    for b in src.iter_mut() {
        *b = (rng.next() & 0xFF) as u8;
    }
    let dst = model::transpose_bytes(&src, iter, w, elem_bytes);
    let nwords = model::num_words(total * elem_bytes) as u32;

    TransposeCase {
        cmd: TransposeCmd {
            bid,
            iter: iter as u32,
            op1_bank,
            wr_bank,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_transpose_i8_roundtrip() {
        let iter = 16;
        let w = 16;
        let src: Vec<u8> = (0..iter * w).map(|i| i as u8).collect();
        let dst = model::transpose_bytes(&src, iter, w, 1);
        for r in 0..iter {
            for c in 0..w {
                assert_eq!(dst[c * iter + r], src[r * w + c]);
            }
        }
    }

    #[test]
    fn model_transpose_i32_roundtrip() {
        let iter = 8;
        let w = 4;
        let src: Vec<u8> = (0..iter * w)
            .map(|i| i as u32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let dst = model::transpose_bytes(&src, iter, w, 4);
        for r in 0..iter {
            for c in 0..w {
                let s = (r * w + c) * 4;
                let d = (c * iter + r) * 4;
                assert_eq!(&dst[d..d + 4], &src[s..s + 4]);
            }
        }
    }

    #[test]
    fn directed_i8_matches_ctest_shape() {
        let case = gen_case(0x1234, 0, 0);
        assert_eq!(case.cmd.bid, 0);
        assert_eq!(case.cmd.iter, 16);
        assert_eq!(case.cmd.elem_bits, 8);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.wr_bank, 1);
        assert_eq!(case.cmd.op1_col, 1);
        assert_eq!(case.cmd.wr_col, 1);
        assert_eq!(case.cmd.num_src_words, 16);
        assert_eq!(case.cmd.num_dst_words, 16);
        assert_eq!(case.src.len(), 256);
        assert_eq!(case.dst.len(), 256);
        assert_eq!(case.src[0], 0);
        assert_eq!(case.src[255], 255);
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(case.dst[c * 16 + r], case.src[r * 16 + c]);
            }
        }
    }

    #[test]
    fn directed_i32_matches_ctest_shape() {
        let case = gen_case(0, 1, 0);
        assert_eq!(case.cmd.iter, 8);
        assert_eq!(case.cmd.elem_bits, 32);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.wr_bank, 1);
        assert_eq!(case.cmd.op1_col, 1);
        assert_eq!(case.cmd.wr_col, 1);
        assert_eq!(case.cmd.num_src_words, 8);
        assert_eq!(case.cmd.num_dst_words, 8);
        assert_eq!(case.src.len(), 128);
        assert_eq!(case.dst.len(), 128);
    }

    #[test]
    fn random_cases_deterministic_and_legal() {
        let a = gen_case(0xCAFE_BABE, 2, 0);
        let b = gen_case(0xCAFE_BABE, 2, 0);
        assert_eq!(a, b);
        assert!(a.cmd.elem_bits == 8 || a.cmd.elem_bits == 32);
        assert!([1u32, 2, 4, 8, 16].contains(&a.cmd.iter));
        assert_eq!(a.cmd.op1_col, 1);
        assert_eq!(a.cmd.wr_col, 1);
        assert_ne!(a.cmd.op1_bank, a.cmd.wr_bank);
        assert!(a.cmd.op1_bank < 32);
        assert!(a.cmd.wr_bank < 32);
        assert!(a.cmd.rob_id < 16);
        assert_eq!(a.cmd.num_src_words, a.cmd.num_dst_words);
        assert_eq!(a.src.len(), a.dst.len());
    }

    #[test]
    fn bid_is_required_arg() {
        let case = gen_case(0, 0, 2);
        assert_eq!(case.cmd.bid, 2);
        let case = gen_case(0, 0, 5);
        assert_eq!(case.cmd.bid, 5);
    }

    #[test]
    fn word_lo_hi_round_trip() {
        let case = gen_case(0x1234, 0, 0);
        for i in 0..case.cmd.num_src_words as usize {
            let lo = case.src_word_lo(i);
            let hi = case.src_word_hi(i);
            let off = i * BANK_ROW_BYTES;
            let expect_lo = u64::from_le_bytes(case.src[off..off + 8].try_into().unwrap());
            let expect_hi = u64::from_le_bytes(case.src[off + 8..off + 16].try_into().unwrap());
            assert_eq!(lo, expect_lo);
            assert_eq!(hi, expect_hi);
        }
    }

    #[test]
    #[should_panic]
    fn word_lo_out_of_range_panics() {
        let case = gen_case(0x1234, 0, 0);
        let _ = case.src_word_lo(MAX_WORDS);
    }
}
