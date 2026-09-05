use crate::model;

pub const MAX_INPUT_WORDS: usize = 64;
pub const MAX_DST: usize = 64;
pub const NUM_CASES: u32 = 9;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxPoolCmd {
    pub bid: u32,
    pub iter: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
    pub op1_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub rs1_lo: u32,
    pub rs1_hi: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub input_base: u32,
    pub output_base: u32,
    pub output_stride: u32,
    pub input_side: u32,
    pub output_side: u32,
    pub kernel: u32,
    pub stride: u32,
    pub padding: u32,
    pub start_row: u32,
    pub start_col: u32,
    pub num_input_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxPoolCase {
    pub cmd: MaxPoolCmd,
    pub input_words: [u128; MAX_INPUT_WORDS],
    pub input_addr: [u32; MAX_INPUT_WORDS],
    pub dst_words: [u128; MAX_DST],
    pub dst_addr: [u32; MAX_DST],
}

impl MaxPoolCase {
    pub fn input_lo(&self, index: usize) -> u64 {
        self.input_words[index] as u64
    }

    pub fn input_hi(&self, index: usize) -> u64 {
        (self.input_words[index] >> 64) as u64
    }

    pub fn dst_lo(&self, index: usize) -> u64 {
        self.dst_words[index] as u64
    }

    pub fn dst_hi(&self, index: usize) -> u64 {
        (self.dst_words[index] >> 64) as u64
    }
}

#[derive(Clone, Copy)]
struct CaseSpec {
    geo: model::Geo,
    op1_bank: u32,
    wr_bank: u32,
    rob_id: u32,
    ctest_input: bool,
    seed: u32,
}

fn field(val: u32, lo: u32, hi: u32) -> u64 {
    let width = hi - lo + 1;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    (u64::from(val) & mask) << lo
}

fn encode_rs1(input_bank: u32, output_bank: u32, iter: u32) -> u64 {
    field(input_bank, 0, 9) | field(0, 10, 19) | field(output_bank, 20, 29) | field(iter, 30, 63)
}

fn encode_rs2(geo: &model::Geo) -> u64 {
    field(geo.input_side as u32, 0, 3)
        | field(geo.output_side as u32, 4, 7)
        | field(geo.kernel as u32, 8, 11)
        | field(geo.stride as u32, 12, 15)
        | field(geo.padding as u32, 16, 19)
        | field(geo.input_base as u32, 20, 25)
        | field(geo.output_base as u32, 26, 31)
        | field(geo.output_stride as u32, 32, 37)
        | field(geo.start_row as u32, 38, 41)
        | field(geo.start_col as u32, 42, 45)
}

fn split_rs(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn validate_cmd(op1_bank: u32, wr_bank: u32, iter: u32, rs1: u64, rs2: u64) {
    if iter == 0 {
        panic!("maxpool: iter must be positive");
    }
    if op1_bank == wr_bank {
        panic!("maxpool: input and output banks must differ");
    }
    if (rs1 & 0x3ff) != u64::from(op1_bank) {
        panic!("maxpool: rs1 bank0 mismatch");
    }
    if ((rs1 >> 10) & 0x3ff) != 0 {
        panic!("maxpool: input bank 1 must be zero");
    }
    if ((rs1 >> 20) & 0x3ff) != u64::from(wr_bank) {
        panic!("maxpool: rs1 bank2 mismatch");
    }
    if ((rs1 >> 30) & 0x3fff_ffff) != u64::from(iter) {
        panic!("maxpool: rs1 iter mismatch");
    }
    if (rs2 >> 46) != 0 {
        panic!("maxpool: reserved rs2 bits must be zero");
    }
}

fn ctest_tile(geo: &model::Geo) -> Vec<u8> {
    let tile = geo.input_side * geo.input_side;
    let mut bytes = vec![0u8; tile * model::BANK_ROW_BYTES];
    for position in 0..tile {
        for channel in 0..model::BANK_ROW_BYTES {
            bytes[position * model::BANK_ROW_BYTES + channel] =
                (((position * 37 + channel * 19) & 255) as i32 - 128) as u8;
        }
    }
    bytes
}

fn seeded_tile(geo: &model::Geo, seed: u32) -> Vec<u8> {
    let tile = geo.input_side * geo.input_side;
    let mut bytes = vec![0u8; tile * model::BANK_ROW_BYTES];
    for position in 0..tile {
        for channel in 0..model::BANK_ROW_BYTES {
            let mix = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(position as u32 * 131)
                .wrapping_add(channel as u32 * 17);
            bytes[position * model::BANK_ROW_BYTES + channel] = ((mix & 255) as i32 - 128) as u8;
        }
    }
    bytes
}

fn copy_words<const N: usize>(src: &[u128]) -> [u128; N] {
    if src.len() > N {
        panic!("maxpool: {} words exceed buffer {N}", src.len());
    }
    let mut out = [0u128; N];
    out[..src.len()].copy_from_slice(src);
    out
}

fn copy_addrs<const N: usize>(src: &[u32]) -> [u32; N] {
    if src.len() > N {
        panic!("maxpool: {} addrs exceed buffer {N}", src.len());
    }
    let mut out = [0u32; N];
    out[..src.len()].copy_from_slice(src);
    out
}

fn build_case(spec: CaseSpec, bid: u32) -> MaxPoolCase {
    model::validate(&spec.geo);
    let iter = model::iter_for(&spec.geo);
    let input_bytes = if spec.ctest_input {
        ctest_tile(&spec.geo)
    } else {
        seeded_tile(&spec.geo, spec.seed)
    };

    let mut referenced = vec![false; spec.geo.input_side * spec.geo.input_side];
    for output_y in 0..spec.geo.output_side {
        for output_x in 0..spec.geo.output_side {
            for kernel_y in 0..spec.geo.kernel {
                for kernel_x in 0..spec.geo.kernel {
                    let y = output_y * spec.geo.stride + kernel_y + spec.geo.start_row;
                    let x = output_x * spec.geo.stride + kernel_x + spec.geo.start_col;
                    if y >= spec.geo.padding
                        && x >= spec.geo.padding
                        && y < spec.geo.padding + spec.geo.input_side
                        && x < spec.geo.padding + spec.geo.input_side
                    {
                        referenced[(y - spec.geo.padding) * spec.geo.input_side
                            + (x - spec.geo.padding)] = true;
                    }
                }
            }
        }
    }

    let mut input_addrs = Vec::new();
    let mut input_words = Vec::new();
    for y in 0..spec.geo.input_side {
        for x in 0..spec.geo.input_side {
            if !referenced[y * spec.geo.input_side + x] {
                continue;
            }
            let addr = (spec.geo.input_base + y * spec.geo.input_side + x) as u32;
            let row_off = (y * spec.geo.input_side + x) * model::BANK_ROW_BYTES;
            let mut row = [0u8; model::BANK_ROW_BYTES];
            row.copy_from_slice(&input_bytes[row_off..row_off + model::BANK_ROW_BYTES]);
            input_addrs.push(addr);
            input_words.push(model::pack_row(&row));
        }
    }

    let gold = model::maxpool_rows(&input_bytes, &spec.geo);
    let dst_addrs: Vec<u32> = gold.iter().map(|(addr, _)| *addr).collect();
    let dst_words: Vec<u128> = gold.iter().map(|(_, row)| model::pack_row(row)).collect();

    if dst_addrs.len() as u32 != iter {
        panic!("maxpool: gold rows {} != iter {iter}", dst_addrs.len());
    }

    let rs1 = encode_rs1(spec.op1_bank, spec.wr_bank, iter);
    let rs2 = encode_rs2(&spec.geo);
    validate_cmd(spec.op1_bank, spec.wr_bank, iter, rs1, rs2);

    let (rs1_lo, rs1_hi) = split_rs(rs1);
    let (rs2_lo, rs2_hi) = split_rs(rs2);

    MaxPoolCase {
        cmd: MaxPoolCmd {
            bid,
            iter,
            op1_bank: spec.op1_bank,
            wr_bank: spec.wr_bank,
            op1_col: 1,
            wr_col: 1,
            rob_id: spec.rob_id,
            rs1_lo,
            rs1_hi,
            rs2_lo,
            rs2_hi,
            input_base: spec.geo.input_base as u32,
            output_base: spec.geo.output_base as u32,
            output_stride: spec.geo.output_stride as u32,
            input_side: spec.geo.input_side as u32,
            output_side: spec.geo.output_side as u32,
            kernel: spec.geo.kernel as u32,
            stride: spec.geo.stride as u32,
            padding: spec.geo.padding as u32,
            start_row: spec.geo.start_row as u32,
            start_col: spec.geo.start_col as u32,
            num_input_words: input_words.len() as u32,
            num_dst_words: iter,
        },
        input_words: copy_words(&input_words),
        input_addr: copy_addrs::<MAX_INPUT_WORDS>(&input_addrs),
        dst_words: copy_words(&dst_words),
        dst_addr: copy_addrs::<MAX_DST>(&dst_addrs),
    }
}

fn spec_for(index: u32) -> CaseSpec {
    match index {
        0 => CaseSpec {
            geo: model::Geo {
                input_side: 6,
                output_side: 3,
                kernel: 2,
                stride: 2,
                padding: 0,
                input_base: 7,
                output_base: 11,
                output_stride: 7,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 2,
            ctest_input: true,
            seed: 0,
        },
        1 => CaseSpec {
            geo: model::Geo {
                input_side: 4,
                output_side: 3,
                kernel: 2,
                stride: 1,
                padding: 0,
                input_base: 0,
                output_base: 20,
                output_stride: 4,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 3,
            ctest_input: false,
            seed: 1,
        },
        2 => CaseSpec {
            geo: model::Geo {
                input_side: 4,
                output_side: 3,
                kernel: 2,
                stride: 1,
                padding: 1,
                input_base: 0,
                output_base: 30,
                output_stride: 4,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 4,
            ctest_input: false,
            seed: 2,
        },
        3 => CaseSpec {
            geo: model::Geo {
                input_side: 6,
                output_side: 3,
                kernel: 2,
                stride: 2,
                padding: 1,
                input_base: 0,
                output_base: 40,
                output_stride: 4,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 5,
            ctest_input: false,
            seed: 3,
        },
        4 => CaseSpec {
            geo: model::Geo {
                input_side: 5,
                output_side: 3,
                kernel: 3,
                stride: 1,
                padding: 0,
                input_base: 0,
                output_base: 50,
                output_stride: 4,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 6,
            ctest_input: false,
            seed: 4,
        },
        5 => CaseSpec {
            geo: model::Geo {
                input_side: 7,
                output_side: 3,
                kernel: 3,
                stride: 2,
                padding: 0,
                input_base: 0,
                output_base: 0,
                output_stride: 7,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 7,
            ctest_input: false,
            seed: 5,
        },
        6 => CaseSpec {
            geo: model::Geo {
                input_side: 5,
                output_side: 3,
                kernel: 3,
                stride: 1,
                padding: 1,
                input_base: 0,
                output_base: 15,
                output_stride: 4,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 8,
            ctest_input: false,
            seed: 6,
        },
        7 => CaseSpec {
            geo: model::Geo {
                input_side: 7,
                output_side: 3,
                kernel: 3,
                stride: 2,
                padding: 1,
                input_base: 0,
                output_base: 25,
                output_stride: 7,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 9,
            ctest_input: false,
            seed: 7,
        },
        8 => CaseSpec {
            geo: model::Geo {
                input_side: 7,
                output_side: 1,
                kernel: 1,
                stride: 1,
                padding: 0,
                input_base: 63,
                output_base: 0,
                output_stride: 1,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 0,
            wr_bank: 1,
            rob_id: 10,
            ctest_input: false,
            seed: 8,
        },
        _ => panic!("maxpool: unsupported directed case {index}"),
    }
}

pub fn gen_case(index: u32, bid: u32) -> MaxPoolCase {
    if index >= NUM_CASES {
        panic!("maxpool: unsupported directed case {index}");
    }
    build_case(spec_for(index), bid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_zero_matches_ctest() {
        let case = gen_case(0, 6);
        assert_eq!(case.cmd.bid, 6);
        assert_eq!(case.cmd.iter, 9);
        assert_eq!(case.cmd.input_side, 6);
        assert_eq!(case.cmd.output_side, 3);
        assert_eq!(case.cmd.kernel, 2);
        assert_eq!(case.cmd.stride, 2);
        assert_eq!(case.cmd.padding, 0);
        assert_eq!(case.cmd.input_base, 7);
        assert_eq!(case.cmd.output_base, 11);
        assert_eq!(case.cmd.output_stride, 7);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.wr_bank, 1);
        assert_eq!(case.cmd.num_input_words, 36);
        assert_eq!(case.cmd.num_dst_words, 9);
        let rs2 = u64::from(case.cmd.rs2_lo) | (u64::from(case.cmd.rs2_hi) << 32);
        assert_eq!(rs2 >> 46, 0);
        let rs1 = u64::from(case.cmd.rs1_lo) | (u64::from(case.cmd.rs1_hi) << 32);
        assert_eq!(rs1 & 0x3ff, 0);
        assert_eq!((rs1 >> 10) & 0x3ff, 0);
        assert_eq!((rs1 >> 20) & 0x3ff, 1);
        assert_eq!((rs1 >> 30) & 0xffff_ffff, 9);
    }

    #[test]
    fn cases_hit_cover_bins() {
        let mut k2 = false;
        let mut k3 = false;
        let mut k1 = false;
        let mut s1 = false;
        let mut s2 = false;
        let mut p0 = false;
        let mut p1 = false;
        for index in 0..NUM_CASES {
            let case = gen_case(index, 6);
            assert_eq!(case.cmd.bid, 6);
            assert_eq!(case.cmd.op1_col, 1);
            assert_eq!(case.cmd.wr_col, 1);
            assert_ne!(case.cmd.op1_bank, case.cmd.wr_bank);
            assert_eq!(case.cmd.num_dst_words, case.cmd.iter);
            assert_eq!(case.cmd.iter, case.cmd.output_side * case.cmd.output_side);
            match case.cmd.kernel {
                1 => k1 = true,
                2 => k2 = true,
                3 => k3 = true,
                other => panic!("unexpected kernel {other}"),
            }
            match case.cmd.stride {
                1 => s1 = true,
                2 => s2 = true,
                other => panic!("unexpected stride {other}"),
            }
            match case.cmd.padding {
                0 => p0 = true,
                1 => p1 = true,
                other => panic!("unexpected padding {other}"),
            }
        }
        assert!(k1 && k2 && k3 && s1 && s2 && p0 && p1);
    }

    #[test]
    #[should_panic(expected = "unsupported directed case")]
    fn unknown_index_panics() {
        let _ = gen_case(NUM_CASES, 6);
    }

    #[test]
    #[should_panic(expected = "illegal square pooling geometry")]
    fn illegal_geometry_panics() {
        model::validate(&model::Geo {
            input_side: 2,
            output_side: 3,
            kernel: 2,
            stride: 1,
            padding: 0,
            input_base: 0,
            output_base: 0,
            output_stride: 4,
            start_row: 0,
            start_col: 0,
        });
    }

    #[test]
    fn gold_matches_manual_window() {
        let geo = model::Geo {
            input_side: 2,
            output_side: 1,
            kernel: 2,
            stride: 1,
            padding: 0,
            input_base: 0,
            output_base: 5,
            output_stride: 1,
            start_row: 0,
            start_col: 0,
        };
        let mut input = vec![0u8; 2 * 2 * model::BANK_ROW_BYTES];
        for lane in 0..model::BANK_ROW_BYTES {
            input[lane] = 1;
            input[model::BANK_ROW_BYTES + lane] = 3;
            input[2 * model::BANK_ROW_BYTES + lane] = 5;
            input[3 * model::BANK_ROW_BYTES + lane] = 2;
        }
        let gold = model::maxpool_rows(&input, &geo);
        assert_eq!(gold.len(), 1);
        assert_eq!(gold[0].0, 5);
        assert_eq!(gold[0].1[0], 5);
    }
}
