use crate::model;

pub const BANK_ROW_BYTES: usize = model::BANK_ROW_BYTES;
pub const MAX_WORDS: usize = 128;
pub const MAX_OUTPUT_ROWS: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Im2colCmd {
    pub bid: u32,
    pub iter: u32,
    pub ksize: u32,
    pub stride: u32,
    pub padding: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
    pub op1_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub num_src_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Im2colCase {
    pub cmd: Im2colCmd,
    pub src: Vec<u8>,
    pub dst: Vec<u8>,
}

impl Im2colCase {
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
    if off + 8 > data.len() {
        panic!("word_lo: word_index {word_index} out of range");
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(buf)
}

fn word_hi(data: &[u8], word_index: usize) -> u64 {
    let off = word_index * BANK_ROW_BYTES + 8;
    if off + 8 > data.len() {
        panic!("word_hi: word_index {word_index} out of range");
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(buf)
}

pub fn gen_case(seed: u32, index: u32, bid: u32) -> Im2colCase {
    match index {
        0 => directed_k3_pad0(bid),
        1 => directed_k3_pad1(bid),
        _ => random_case(seed, index, bid),
    }
}

fn build_case(
    bid: u32,
    iter: usize,
    ksize: usize,
    stride: usize,
    padding: usize,
    op1_bank: u32,
    wr_bank: u32,
    rob_id: u32,
    flat: &[u8],
) -> Im2colCase {
    if op1_bank == wr_bank {
        panic!("build_case: op1_bank and wr_bank must differ");
    }
    if op1_bank > 7 || wr_bank > 7 {
        panic!("build_case: banks must be in 0..7");
    }
    if flat.len() != iter * iter {
        panic!("build_case: flat len mismatch");
    }
    let rows = model::output_rows(iter, ksize, stride, padding);
    if rows > MAX_OUTPUT_ROWS {
        panic!("build_case: output_rows={rows} exceeds {MAX_OUTPUT_ROWS}");
    }
    let dst_flat = model::im2col(flat, iter, ksize, stride, padding);
    let src = model::pack_bank_words(flat);
    let nsrc = model::num_words(src.len());
    let ndst = model::num_words(dst_flat.len());
    if nsrc > MAX_WORDS || ndst > MAX_WORDS {
        panic!("build_case: word count out of range src={nsrc} dst={ndst}");
    }
    Im2colCase {
        cmd: Im2colCmd {
            bid,
            iter: iter as u32,
            ksize: ksize as u32,
            stride: stride as u32,
            padding: padding as u32,
            op1_bank,
            wr_bank,
            op1_col: 1,
            wr_col: 1,
            rob_id,
            num_src_words: nsrc as u32,
            num_dst_words: ndst as u32,
        },
        src,
        dst: dst_flat,
    }
}

fn directed_k3_pad0(bid: u32) -> Im2colCase {
    let iter = 6usize;
    let k = 3usize;
    let flat: [i8; 36] = [
        -1, 2, 3, 4, -5, 6, 0, 8, 9, -10, 11, 12, 13, 0, -15, 16, 17, 18, 19, -20, 0, 22, 23, -24,
        -25, 26, 27, 0, -29, 30, 31, 32, 33, -34, 0, 36,
    ];
    let bytes: Vec<u8> = flat.iter().map(|&x| x as u8).collect();
    build_case(bid, iter, k, 1, 0, 0, 1, 3, &bytes)
}

fn directed_k3_pad1(bid: u32) -> Im2colCase {
    let iter = 6usize;
    let k = 3usize;
    let flat: [i8; 36] = [
        7, -2, 3, 0, 5, -6, 1, 8, -9, 10, 11, 0, -4, 12, 13, 14, 0, 16, 17, 0, 19, -20, 21, 22, 0,
        24, 25, 26, -27, 28, 29, -30, 0, 32, 33, 34,
    ];
    let bytes: Vec<u8> = flat.iter().map(|&x| x as u8).collect();
    build_case(bid, iter, k, 1, 1, 0, 1, 5, &bytes)
}

fn shape_ok(iter: usize, ksize: usize, stride: usize, padding: usize) -> bool {
    if iter == 0 || ksize == 0 || stride == 0 {
        return false;
    }
    let padded = iter + 2 * padding;
    if padded < ksize {
        return false;
    }
    let rows = model::output_rows(iter, ksize, stride, padding);
    rows <= MAX_OUTPUT_ROWS
}

fn random_case(seed: u32, index: u32, bid: u32) -> Im2colCase {
    let mut rng = Rng::new(seed, index);
    let iter_pool = [3usize, 4, 5, 6, 7];
    let ksize_pool = [1usize, 3, 5];
    let pad_pool = [0usize, 1];
    let base = (index - 2) as usize;

    for attempt in 0..10_000 {
        let iter = iter_pool[(base + attempt) % iter_pool.len()];
        let ksize = ksize_pool[((base / iter_pool.len()) + attempt) % ksize_pool.len()];
        let stride = 1usize;
        let padding = pad_pool[(base + attempt / 3) % pad_pool.len()];
        if !shape_ok(iter, ksize, stride, padding) {
            continue;
        }
        let op1_bank = rng.range(0, 7);
        let mut wr_bank = rng.range(0, 7);
        if wr_bank == op1_bank {
            wr_bank = (wr_bank + 1) & 7;
        }
        let mut flat = vec![0u8; iter * iter];
        for b in flat.iter_mut() {
            *b = (rng.next() & 0xff) as u8;
        }
        return build_case(
            bid,
            iter,
            ksize,
            stride,
            padding,
            op1_bank,
            wr_bank,
            rng.range(0, 15),
            &flat,
        );
    }
    panic!("random_case: failed to find legal shape for index={index}");
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
    fn directed_k3_pad0_matches_ctest() {
        let case = gen_case(0, 0, 2);
        assert_eq!(case.cmd.iter, 6);
        assert_eq!(case.cmd.ksize, 3);
        assert_eq!(case.cmd.stride, 1);
        assert_eq!(case.cmd.padding, 0);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.wr_bank, 1);
        assert_eq!(case.cmd.op1_col, 1);
        assert_eq!(case.cmd.wr_col, 1);
        assert_eq!(case.cmd.num_src_words, 3);
        assert_eq!(case.cmd.num_dst_words, 16);
        assert_eq!(case.src.len(), 48);
        assert_eq!(case.dst.len(), 16 * 16);
        assert_eq!(case.src[0], (-1i8) as u8);
        assert_eq!(case.src[35], 36);
    }

    #[test]
    fn directed_k3_pad1_shape() {
        let case = gen_case(0, 1, 2);
        assert_eq!(case.cmd.iter, 6);
        assert_eq!(case.cmd.ksize, 3);
        assert_eq!(case.cmd.padding, 1);
        assert_eq!(case.cmd.num_dst_words, 48);
    }

    #[test]
    fn random_deterministic_and_legal() {
        let a = gen_case(0xCAFE_BABE, 2, 2);
        let b = gen_case(0xCAFE_BABE, 2, 2);
        assert_eq!(a, b);
        assert!([3u32, 4, 5, 6, 7].contains(&a.cmd.iter));
        assert!([1u32, 3, 5].contains(&a.cmd.ksize));
        assert_eq!(a.cmd.stride, 1);
        assert!(a.cmd.padding == 0 || a.cmd.padding == 1);
        assert_ne!(a.cmd.op1_bank, a.cmd.wr_bank);
        assert!(a.cmd.num_dst_words <= MAX_OUTPUT_ROWS as u32);
    }

    #[test]
    fn coverage_bins_hit_across_suite() {
        let mut saw_iter = [false; 8];
        let mut saw_k = [false; 6];
        let mut saw_pad = [false; 2];
        for i in 0..=21 {
            let c = gen_case(0xCAFE_BABE, i, 2);
            saw_iter[c.cmd.iter as usize] = true;
            saw_k[c.cmd.ksize as usize] = true;
            saw_pad[c.cmd.padding as usize] = true;
            assert_eq!(c.cmd.stride, 1);
        }
        for it in [3usize, 4, 5, 6, 7] {
            assert!(saw_iter[it], "missing iter bin {it}");
        }
        for k in [1usize, 3, 5] {
            assert!(saw_k[k], "missing ksize bin {k}");
        }
        assert!(saw_pad[0] && saw_pad[1], "missing pad bins");
    }

    #[test]
    fn bid_required_arg() {
        assert_eq!(gen_case(0, 0, 2).cmd.bid, 2);
        assert_eq!(gen_case(0, 0, 3).cmd.bid, 3);
    }

    #[test]
    fn word_lo_hi_round_trip() {
        let case = gen_case(0, 0, 2);
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
        let case = gen_case(0, 0, 2);
        let _ = case.src_word_lo(MAX_WORDS);
    }
}
