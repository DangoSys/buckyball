pub const WORDS: usize = 4;
pub const GROUPS: usize = 4;
pub const MAX_WORDS: usize = WORDS * GROUPS;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToInt8Cmd {
    pub bid: u32,
    pub iter: u32,
    pub da_bits: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
    pub op1_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub num_src_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToInt8Case {
    pub cmd: ToInt8Cmd,
    pub input_words: [u128; MAX_WORDS],
}

impl ToInt8Case {
    pub fn word_lo(&self, index: usize) -> u64 {
        self.input_words[index] as u64
    }

    pub fn word_hi(&self, index: usize) -> u64 {
        (self.input_words[index] >> 64) as u64
    }
}

pub fn gen_case(index: u32, bid: u32) -> ToInt8Case {
    match index {
        0 => directed_i8_case(bid),
        1 => zero_i8_case(bid),
        2 => rounding_i8_case(bid),
        3 => rows_i8_case(bid),
        4 => scale_rows_i8_case(bid),
        5 => stream_1x4_case(bid),
        6 => stream_2x2_case(bid),
        7 => stream_8x1_case(bid),
        _ => panic!("toint8: unsupported directed case {index}"),
    }
}

fn directed_i8_case(bid: u32) -> ToInt8Case {
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

    ToInt8Case {
        cmd: ToInt8Cmd {
            bid,
            iter: 1,
            da_bits: 0,
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

fn zero_i8_case(bid: u32) -> ToInt8Case {
    ToInt8Case {
        cmd: ToInt8Cmd {
            bid,
            iter: 1,
            da_bits: 0,
            op1_bank: 0,
            wr_bank: 1,
            op1_col: 4,
            wr_col: 1,
            rob_id: 3,
            num_src_words: GROUPS as u32,
        },
        input_words: [0; MAX_WORDS],
    }
}

fn rounding_i8_case(bid: u32) -> ToInt8Case {
    let vals: [u32; 16] = [
        0x3f00_0000, 0xbf00_0000, 0x3fc0_0000, 0xbfc0_0000,
        0x4020_0000, 0xc020_0000, 0x4060_0000, 0xc060_0000,
        0x4090_0000, 0xc090_0000, 0x40b0_0000, 0xc0b0_0000,
        0x42fd_0000, 0xc2fd_0000, 0x42fe_0000, 0xc2fe_0000,
    ];
    let mut input_words = [0u128; MAX_WORDS];
    for group in 0..GROUPS {
        for lane in 0..4 {
            input_words[group] |= u128::from(vals[group * 4 + lane]) << (lane * 32);
        }
    }
    ToInt8Case {
        cmd: ToInt8Cmd {
            bid, iter: 1, da_bits: 0, op1_bank: 0, wr_bank: 1,
            op1_col: 4, wr_col: 1, rob_id: 4, num_src_words: GROUPS as u32,
        },
        input_words,
    }
}

fn rows_i8_case(bid: u32) -> ToInt8Case {
    let vals: [u32; 32] = [
        0xc100_0000, 0xc0e0_0000, 0xc0c0_0000, 0xc0a0_0000,
        0xc080_0000, 0xc040_0000, 0xc000_0000, 0xbf80_0000,
        0x0000_0000, 0x42fe_0000, 0x3f80_0000, 0x4000_0000,
        0x4040_0000, 0x4080_0000, 0x40a0_0000, 0x40c0_0000,
        0x40e0_0000, 0x4100_0000, 0xc100_0000, 0xc0e0_0000,
        0xc0c0_0000, 0xc0a0_0000, 0xc080_0000, 0xc040_0000,
        0xc000_0000, 0xbf80_0000, 0x0000_0000, 0x3f80_0000,
        0x4000_0000, 0x4040_0000, 0x4080_0000, 0x40a0_0000,
    ];
    let mut input_words = [0u128; MAX_WORDS];
    for row in 0..2 {
        for group in 0..GROUPS {
            for lane in 0..4 {
                let index = row * 16 + group * 4 + lane;
                input_words[row * GROUPS + group] |= u128::from(vals[index]) << (lane * 32);
            }
        }
    }
    ToInt8Case {
        cmd: ToInt8Cmd {
            bid, iter: 2, da_bits: 0, op1_bank: 0, wr_bank: 1,
            op1_col: 4, wr_col: 1, rob_id: 5, num_src_words: 8,
        },
        input_words,
    }
}

fn scale_rows_i8_case(bid: u32) -> ToInt8Case {
    let vals: [u32; 32] = [
        0x3f80_0000, 0xbf80_0000, 0x4000_0000, 0xc000_0000,
        0x4040_0000, 0xc040_0000, 0x4080_0000, 0xc080_0000,
        0x40a0_0000, 0xc0a0_0000, 0x40c0_0000, 0xc0c0_0000,
        0x40e0_0000, 0xc0e0_0000, 0x4100_0000, 0xc100_0000,
        0x41a0_0000, 0xc1a0_0000, 0x4120_0000, 0xc120_0000,
        0x4170_0000, 0xc170_0000, 0x40a0_0000, 0xc0a0_0000,
        0x3f80_0000, 0xbf80_0000, 0x4000_0000, 0xc000_0000,
        0x4040_0000, 0xc040_0000, 0x4080_0000, 0xc080_0000,
    ];
    let mut input_words = [0u128; MAX_WORDS];
    for row in 0..2 {
        for group in 0..GROUPS {
            for lane in 0..4 {
                let index = row * 16 + group * 4 + lane;
                input_words[row * GROUPS + group] |= u128::from(vals[index]) << (lane * 32);
            }
        }
    }
    ToInt8Case {
        cmd: ToInt8Cmd {
            bid, iter: 2, da_bits: 0, op1_bank: 0, wr_bank: 1,
            op1_col: 4, wr_col: 1, rob_id: 6, num_src_words: 8,
        },
        input_words,
    }
}

fn stream_1x4_case(bid: u32) -> ToInt8Case {
    let mut case = directed_i8_case(bid);
    case.cmd.iter = 4;
    case.cmd.op1_col = 1;
    case.cmd.rob_id = 7;
    case
}

fn stream_2x2_case(bid: u32) -> ToInt8Case {
    let mut case = directed_i8_case(bid);
    case.cmd.iter = 2;
    case.cmd.op1_col = 2;
    case.cmd.rob_id = 8;
    case
}

fn stream_8x1_case(bid: u32) -> ToInt8Case {
    let mut case = rows_i8_case(bid);
    case.cmd.iter = 1;
    case.cmd.op1_col = 8;
    case.cmd.rob_id = 9;
    case
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_cases_use_online_i8_layout() {
        let case = gen_case(0, 3);

        assert_eq!(case.cmd.bid, 3);
        assert_eq!(case.cmd.iter, 1);
        assert_eq!(case.cmd.da_bits, 0);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.wr_bank, 1);
        assert_eq!(case.cmd.op1_col, 4);
        assert_eq!(case.cmd.wr_col, 1);
        assert_eq!(case.cmd.rob_id, 2);
        assert_eq!(case.cmd.num_src_words, GROUPS as u32);
    }

    #[test]
    fn case_one_is_zero_i8_case() {
        let case = gen_case(1, 5);
        assert_eq!(case.cmd.bid, 5);
        assert_eq!(case.cmd.iter, 1);
        assert_eq!(case.cmd.da_bits, 0);
        assert_eq!(case.input_words, [0; MAX_WORDS]);
    }

    #[test]
    fn bid_is_required_arg() {
        let case = gen_case(0, 3);
        assert_eq!(case.cmd.bid, 3);
        let case = gen_case(0, 5);
        assert_eq!(case.cmd.bid, 5);
    }

    #[test]
    fn rounding_and_rows_cases_cover_all_words() {
        assert_eq!(gen_case(2, 3).cmd.iter, 1);
        assert_eq!(gen_case(3, 3).cmd.iter, 2);
        assert_eq!(gen_case(4, 3).cmd.iter, 2);
    }

    #[test]
    fn stream_cases_cover_independent_group_counts() {
        assert_eq!((gen_case(5, 3).cmd.iter, gen_case(5, 3).cmd.op1_col), (4, 1));
        assert_eq!((gen_case(6, 3).cmd.iter, gen_case(6, 3).cmd.op1_col), (2, 2));
        assert_eq!((gen_case(7, 3).cmd.iter, gen_case(7, 3).cmd.op1_col), (1, 8));
    }
}
