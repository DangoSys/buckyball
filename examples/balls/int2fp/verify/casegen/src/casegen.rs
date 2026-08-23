pub const GROUPS: usize = 4;
pub const MAX_ITER: usize = 16;
pub const MAX_WORDS: usize = MAX_ITER * GROUPS;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int2FpCmd {
    pub bid: u32,
    pub iter: u32,
    pub da_bits: u32,
    pub dw_addr: u32,
    pub dw_bits: u32,
    pub per_channel: u32,
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
    pub channel_dw_bits: [u32; GROUPS * 4],
}

impl Int2FpCase {
    pub fn word_lo(&self, index: usize) -> u64 {
        self.input_words[index] as u64
    }

    pub fn word_hi(&self, index: usize) -> u64 {
        (self.input_words[index] >> 64) as u64
    }

    pub fn dw_bits(&self, index: usize) -> u32 {
        self.channel_dw_bits[index]
    }
}

pub fn gen_case(index: u32, bid: u32) -> Int2FpCase {
    match index {
        0 => directed_tensor(bid),
        1 => directed_channel(bid),
        2 => tensor_groups(bid),
        3 => channel_base(bid),
        4 => channel_two_rows(bid),
        _ => panic!("int2fp: unsupported directed case {index}"),
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

fn directed_tensor(bid: u32) -> Int2FpCase {
    let vals: [i32; 64] = [
        1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8, 1, 2, 3, -1, -2, 0, 4, 5, 10,
        -10, 7, 100, -100, 8, 16, -8, 1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
        1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
    ];
    let iter = 16usize;
    let mut input_words = [0u128; MAX_WORDS];
    for w in 0..iter {
        input_words[w] = pack_i32s(&vals[w * 4..w * 4 + 4]);
    }

    Int2FpCase {
        cmd: Int2FpCmd {
            bid,
            iter: iter as u32,
            da_bits: 0x3F00_0000,
            dw_addr: 16,
            dw_bits: 0x3E80_0000,
            per_channel: 0,
            op1_bank: 0,
            wr_bank: 1,
            op1_col: 1,
            wr_col: 1,
            rob_id: 3,
            num_src_words: iter as u32,
        },
        input_words,
        channel_dw_bits: [0x3E80_0000; GROUPS * 4],
    }
}

fn directed_channel(bid: u32) -> Int2FpCase {
    directed_channel_at(bid, 16, 3)
}

fn directed_channel_at(bid: u32, dw_addr: u32, rob_id: u32) -> Int2FpCase {
    let mut input_words = [0u128; MAX_WORDS];
    for group in 0..GROUPS {
        input_words[group] = pack_i32s(&[8, -8, 16, -16]);
    }
    let channel_dw_bits = [
        0x3e80_0000, 0x3f00_0000, 0x3f40_0000, 0x3f80_0000,
        0x3fa0_0000, 0x3fc0_0000, 0x3fe0_0000, 0x4000_0000,
        0x3e00_0000, 0x3e80_0000, 0x3ec0_0000, 0x3f00_0000,
        0x4010_0000, 0x4020_0000, 0x4030_0000, 0x4040_0000,
    ];
    Int2FpCase {
        cmd: Int2FpCmd {
            bid,
            iter: 1,
            da_bits: 0x3f00_0000,
            dw_addr,
            dw_bits: channel_dw_bits[0],
            per_channel: 1,
            op1_bank: 0,
            wr_bank: 1,
            op1_col: GROUPS as u32,
            wr_col: GROUPS as u32,
            rob_id,
            num_src_words: GROUPS as u32,
        },
        input_words,
        channel_dw_bits,
    }
}

fn tensor_groups(bid: u32) -> Int2FpCase {
    let mut input_words = [0u128; MAX_WORDS];
    input_words[0] = pack_i32s(&[2, -2, 4, -4]);
    input_words[1] = pack_i32s(&[6, -6, 8, -8]);
    input_words[2] = pack_i32s(&[10, -10, 12, -12]);
    input_words[3] = pack_i32s(&[14, -14, 16, -16]);
    Int2FpCase {
        cmd: Int2FpCmd {
            bid,
            iter: 1,
            da_bits: 0x3f00_0000,
            dw_addr: 16,
            dw_bits: 0x3e80_0000,
            per_channel: 0,
            op1_bank: 0,
            wr_bank: 1,
            op1_col: GROUPS as u32,
            wr_col: GROUPS as u32,
            rob_id: 4,
            num_src_words: GROUPS as u32,
        },
        input_words,
        channel_dw_bits: [0x3e80_0000; GROUPS * 4],
    }
}

fn channel_base(bid: u32) -> Int2FpCase {
    directed_channel_at(bid, 64, 5)
}

fn channel_two_rows(bid: u32) -> Int2FpCase {
    let mut case = directed_channel(bid);
    case.cmd.iter = 2;
    case.cmd.rob_id = 6;
    case.cmd.num_src_words = 8;
    for group in 0..GROUPS {
        case.input_words[group] = pack_i32s(&[8, 8, 8, 8]);
        case.input_words[GROUPS + group] = pack_i32s(&[16, 16, 16, 16]);
    }
    case
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model;

    #[test]
    fn case_zero_is_fp32_smoke() {
        let case = gen_case(0, 4);
        assert_eq!(case.cmd.bid, 4);
        assert_eq!(case.cmd.iter, 16);
        assert_eq!(case.cmd.da_bits, 0x3F00_0000);
        assert_eq!(case.cmd.dw_addr, 16);
        assert_eq!(case.cmd.dw_bits, 0x3E80_0000);
        assert_eq!(case.cmd.per_channel, 0);
        assert_eq!(case.cmd.op1_col, 1);
        assert_eq!(case.cmd.wr_col, 1);
        assert_eq!(case.cmd.num_src_words, 16);
        assert_eq!(
            model::int2fp_dequant_bits(8, 0x3F00_0000, 0x3E80_0000),
            1.0f32.to_bits()
        );
    }

    #[test]
    fn case_one_is_fp32() {
        let case = gen_case(1, 6);
        assert_eq!(case.cmd.bid, 6);
        assert_eq!(case.cmd.op1_col, GROUPS as u32);
        assert_eq!(case.cmd.wr_col, GROUPS as u32);
        assert_eq!(case.cmd.per_channel, 1);
        assert_eq!(case.cmd.dw_addr, 16);
        assert_eq!(case.cmd.num_src_words, GROUPS as u32);
    }

    #[test]
    fn bid_is_required_arg() {
        assert_eq!(gen_case(0, 4).cmd.bid, 4);
        assert_eq!(gen_case(0, 6).cmd.bid, 6);
    }

    #[test]
    fn channel_base_uses_nonzero_dw_address() {
        let case = gen_case(3, 4);
        assert_eq!(case.cmd.dw_addr, 64);
        assert_eq!(case.cmd.per_channel, 1);
    }

    #[test]
    fn channel_two_rows_reuses_all_channel_scales() {
        let case = gen_case(4, 4);
        assert_eq!(case.cmd.iter, 2);
        assert_eq!(case.cmd.num_src_words, 8);
        assert_eq!(case.cmd.op1_col, GROUPS as u32);
    }
}
