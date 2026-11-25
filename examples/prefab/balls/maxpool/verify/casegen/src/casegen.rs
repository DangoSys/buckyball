use crate::model;

pub const MAX_INPUT_WORDS: usize = 128;
pub const MAX_DST: usize = 256;
pub const NUM_CASES: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxPoolCmd {
    pub is_topk: u32,
    pub iter: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
    pub op1_group: u32,
    pub wr_group: u32,
    pub op1_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
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
    // Channel group of each word, relative to the command's base group.
    pub input_group: [u32; MAX_INPUT_WORDS],
    pub dst_words: [u128; MAX_DST],
    pub dst_addr: [u32; MAX_DST],
    pub dst_group: [u32; MAX_DST],
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
    // Channel groups of the input and output banks; group g is pooled with
    // the same geometry after group g - 1.
    groups: u32,
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

fn validate_cmd(op1_bank: u32, wr_bank: u32, iter: u32, rs2: u64) {
    if iter == 0 {
        panic!("maxpool: iter must be positive");
    }
    if op1_bank == wr_bank {
        panic!("maxpool: input and output banks must differ");
    }
    if (rs2 >> 46) != 0 {
        panic!("maxpool: reserved rs2 bits must be zero");
    }
}

fn ctest_tile(geo: &model::Geo) -> Vec<u8> {
    let hart = geo.input_side * geo.input_side;
    let mut bytes = vec![0u8; hart * model::BANK_ROW_BYTES];
    for position in 0..hart {
        for channel in 0..model::BANK_ROW_BYTES {
            bytes[position * model::BANK_ROW_BYTES + channel] =
                (((position * 37 + channel * 19) & 255) as i32 - 128) as u8;
        }
    }
    bytes
}

fn seeded_tile(geo: &model::Geo, seed: u32) -> Vec<u8> {
    let hart = geo.input_side * geo.input_side;
    let mut bytes = vec![0u8; hart * model::BANK_ROW_BYTES];
    for position in 0..hart {
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

fn build_case(spec: CaseSpec) -> MaxPoolCase {
    model::validate(&spec.geo);
    let iter = model::iter_for(&spec.geo);

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
    let mut input_groups = Vec::new();
    let mut dst_addrs = Vec::new();
    let mut dst_words = Vec::new();
    let mut dst_groups = Vec::new();
    for group in 0..spec.groups {
        let input_bytes = if spec.ctest_input {
            ctest_tile(&spec.geo)
        } else {
            seeded_tile(&spec.geo, spec.seed + group * 101)
        };
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
                input_groups.push(group);
            }
        }
        for (addr, row) in model::maxpool_rows(&input_bytes, &spec.geo) {
            dst_addrs.push(addr);
            dst_words.push(model::pack_row(&row));
            dst_groups.push(group);
        }
    }

    if dst_addrs.len() as u32 != iter * spec.groups {
        panic!(
            "maxpool: gold rows {} != iter {iter} x groups",
            dst_addrs.len()
        );
    }

    let rs2 = encode_rs2(&spec.geo);
    validate_cmd(spec.op1_bank, spec.wr_bank, iter, rs2);

    let (rs2_lo, rs2_hi) = split_rs(rs2);

    MaxPoolCase {
        cmd: MaxPoolCmd {
            is_topk: 0,
            iter,
            op1_bank: spec.op1_bank,
            wr_bank: spec.wr_bank,
            // Vary the renamed group bases across cases; the Ball adds them to every access.
            op1_group: spec.seed % 3,
            wr_group: (spec.seed + 1) % 4,
            op1_col: spec.groups,
            wr_col: spec.groups,
            rob_id: spec.rob_id,
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
            num_dst_words: dst_words.len() as u32,
        },
        input_words: copy_words(&input_words),
        input_addr: copy_addrs::<MAX_INPUT_WORDS>(&input_addrs),
        input_group: copy_addrs::<MAX_INPUT_WORDS>(&input_groups),
        dst_words: copy_words(&dst_words),
        dst_addr: copy_addrs::<MAX_DST>(&dst_addrs),
        dst_group: copy_addrs::<MAX_DST>(&dst_groups),
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
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
            groups: 1,
        },
        // Two channel groups, 3x3 kernel with padding.
        9 => CaseSpec {
            geo: model::Geo {
                input_side: 5,
                output_side: 3,
                kernel: 3,
                stride: 2,
                padding: 1,
                input_base: 2,
                output_base: 5,
                output_stride: 4,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 2,
            wr_bank: 5,
            rob_id: 11,
            ctest_input: false,
            seed: 9,
            groups: 2,
        },
        // Three channel groups, 2x2 stride-2 pooling.
        10 => CaseSpec {
            geo: model::Geo {
                input_side: 4,
                output_side: 2,
                kernel: 2,
                stride: 2,
                padding: 0,
                input_base: 0,
                output_base: 0,
                output_stride: 2,
                start_row: 0,
                start_col: 0,
            },
            op1_bank: 1,
            wr_bank: 0,
            rob_id: 12,
            ctest_input: false,
            seed: 10,
            groups: 3,
        },
        _ => panic!("maxpool: unsupported directed case {index}"),
    }
}

pub fn gen_case(index: u32) -> MaxPoolCase {
    if index >= NUM_CASES {
        panic!("maxpool: unsupported directed case {index}");
    }
    if index < 11 {
        build_case(spec_for(index))
    } else {
        build_topk(index - 11)
    }
}

fn build_topk(index: u32) -> MaxPoolCase {
    let (rows, groups, n, k) = match index {
        0 => (5, 2, 37, 9),
        1 => (2, 1, 7, 7),
        2 => (2, 3, 19, 1),
        3 => (76, 1, 304, 300),
        4 => (16, 2, 127, 32),
        _ => panic!("unknown Top-K case {index}"),
    };
    let mut keys = vec![u32::MAX; rows * groups * 4];
    for (i, key) in keys.iter_mut().take(n).enumerate() {
        *key = match index {
            0 => match i % 5 {
                0 => u32::MAX,
                1 => 0x8000_0000,
                3 => 0,
                _ => 17,
            },
            1 | 3 => 0,
            2 => 0x8000_0000 + i as u32,
            4 => (i as u32).wrapping_mul(0x9e37_79b9) ^ 0x6a09_e667,
            _ => unreachable!(),
        };
    }
    let mut input_words = [0u128; MAX_INPUT_WORDS];
    let mut input_addr = [0u32; MAX_INPUT_WORDS];
    let mut input_group = [0u32; MAX_INPUT_WORDS];
    for row in 0..rows {
        for group in 0..groups {
            let word = row * groups + group;
            for lane in 0..4 {
                input_words[word] |= (keys[word * 4 + lane] as u128) << (lane * 32);
            }
            input_addr[word] = row as u32;
            input_group[word] = group as u32;
        }
    }
    // Independent selection reference: mark chosen positions, keeping the
    // first index on a tie. The BEMU model uses a full lexicographic sort.
    let mut used = vec![false; n];
    let mut dst_words = [0u128; MAX_DST];
    let mut dst_addr = [0u32; MAX_DST];
    for rank in 0..k {
        let mut best = None;
        for at in 0..n {
            if !used[at] && best.is_none_or(|old| keys[at] > keys[old]) {
                best = Some(at);
            }
        }
        let at = best.unwrap();
        used[at] = true;
        let pair = keys[at] as u128 | ((at as u128) << 32);
        dst_words[rank / 2] |= pair << ((rank % 2) * 64);
        dst_addr[rank / 2] = (rank / 2) as u32;
    }
    MaxPoolCase {
        cmd: MaxPoolCmd {
            is_topk: 1,
            iter: rows as u32,
            op1_bank: 0,
            wr_bank: 1,
            op1_group: index % 3,
            wr_group: (index + 1) % 4,
            op1_col: groups as u32,
            wr_col: 1,
            rob_id: index + 2,
            rs2_lo: n as u32,
            rs2_hi: k as u32,
            input_base: 0,
            output_base: 0,
            output_stride: 0,
            input_side: 0,
            output_side: 0,
            kernel: 0,
            stride: 0,
            padding: 0,
            start_row: 0,
            start_col: 0,
            num_input_words: (rows * groups) as u32,
            num_dst_words: k.div_ceil(2) as u32,
        },
        input_words,
        input_addr,
        input_group,
        dst_words,
        dst_addr,
        dst_group: [0; MAX_DST],
    }
}

#[cfg(test)]
mod tests {
    use super::gen_case;

    #[test]
    fn topk_matches_c_vectors_and_crosses_eight_bit_rank() {
        let case = gen_case(11);
        for rank in 0..9usize {
            let pair = case.dst_words[rank / 2] >> ((rank % 2) * 64);
            assert_eq!(pair as u32, if rank < 8 { u32::MAX } else { 0x8000_0000 });
            assert_eq!(
                (pair >> 32) as u32,
                if rank < 8 { rank as u32 * 5 } else { 1 }
            );
        }
        assert_eq!(case.dst_words[4] >> 64, 0);
        let large = gen_case(14);
        assert_eq!(large.cmd.rs2_hi, 300);
        assert_eq!(large.cmd.num_dst_words, 150);
        assert_eq!(large.dst_words[149], (298u128 << 32) | (299u128 << 96));
        for index in 0..super::NUM_CASES {
            let case = gen_case(index);
            assert!(case.cmd.num_input_words as usize <= super::MAX_INPUT_WORDS);
            assert!(case.cmd.num_dst_words as usize <= super::MAX_DST);
        }
    }
}
