pub const BYTES_PER_ROW: usize = 16;
pub const SHARED_TABLE_BYTES: usize = 256;
pub const LANE_TABLE_BYTES: usize = 16 * SHARED_TABLE_BYTES;
pub const MAX_SRC: usize = 8;
pub const MAX_LUT: usize = LANE_TABLE_BYTES / BYTES_PER_ROW;
pub const MAX_DST: usize = 8;
pub const NUM_CASES: u32 = 10;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LutCmd {
    pub mode: u32,
    pub multiplier: u32,
    pub iter: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub op1_group: u32,
    pub op2_group: u32,
    pub wr_group: u32,
    pub op1_col: u32,
    pub op2_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub num_src_words: u32,
    pub num_lut_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutCase {
    pub cmd: LutCmd,
    pub src_words: [u128; MAX_SRC],
    pub lut_words: [u128; MAX_LUT],
    pub dst_words: [u128; MAX_DST],
}

impl LutCase {
    pub fn src_lo(&self, index: usize) -> u64 {
        self.src_words[index] as u64
    }

    pub fn src_hi(&self, index: usize) -> u64 {
        (self.src_words[index] >> 64) as u64
    }

    pub fn lut_lo(&self, index: usize) -> u64 {
        self.lut_words[index] as u64
    }

    pub fn lut_hi(&self, index: usize) -> u64 {
        (self.lut_words[index] >> 64) as u64
    }

    pub fn dst_lo(&self, index: usize) -> u64 {
        self.dst_words[index] as u64
    }

    pub fn dst_hi(&self, index: usize) -> u64 {
        (self.dst_words[index] >> 64) as u64
    }
}

pub fn gen_case(index: u32) -> LutCase {
    if index >= NUM_CASES {
        panic!("lut: unsupported directed case {index}");
    }
    match index {
        0 => directed(2, 4, false, [0, 0, 0]),
        1 => directed(3, 1, false, [1, 2, 3]),
        2 => directed(4, 8, false, [2, 0, 1]),
        3 => directed(5, 4, true, [3, 1, 2]),
        4..=9 => wide(index),
        _ => panic!("lut: unsupported directed case {index}"),
    }
}

fn shared_table() -> Vec<u8> {
    let mut table = vec![0u8; SHARED_TABLE_BYTES];
    for (i, slot) in table.iter_mut().enumerate() {
        *slot = (((i * 73 + 19) ^ 0xa5) & 0xff) as u8;
    }
    table
}

fn lane_table() -> Vec<u8> {
    let mut table = vec![0u8; LANE_TABLE_BYTES];
    for channel in 0..16 {
        for input in 0..256 {
            table[channel * 256 + input] = ((input as i32 + channel as i32 * 17 - 128) as i8) as u8;
        }
    }
    table
}

fn pack_lane_table(table: &[u8]) -> Vec<u8> {
    if table.len() != LANE_TABLE_BYTES {
        panic!("lut: invalid lane table size {}", table.len());
    }
    let mut packed = vec![0u8; LANE_TABLE_BYTES];
    for row in 0..64 {
        for group in 0..4 {
            for byte in 0..16 {
                packed[group * 64 * 16 + row * 16 + byte] =
                    table[(group * 4 + row / 16) * 256 + (row % 16) * 16 + byte];
            }
        }
    }
    packed
}

fn ctest_input(nbytes: usize) -> Vec<u8> {
    (0..nbytes)
        .map(|i| (i as i32 * 29 - 128) as i8 as u8)
        .collect()
}

fn pack_bytes(bytes: &[u8]) -> Vec<u128> {
    if bytes.len() % BYTES_PER_ROW != 0 {
        panic!(
            "lut: {} bytes is not a multiple of {BYTES_PER_ROW}",
            bytes.len()
        );
    }
    bytes
        .chunks(BYTES_PER_ROW)
        .map(|chunk| {
            let mut word = 0u128;
            for (lane, byte) in chunk.iter().enumerate() {
                word |= u128::from(*byte) << (lane * 8);
            }
            word
        })
        .collect()
}

fn copy_words<const N: usize>(src: &[u128]) -> [u128; N] {
    if src.len() > N {
        panic!("lut: {} words exceed buffer {N}", src.len());
    }
    let mut out = [0u128; N];
    out[..src.len()].copy_from_slice(src);
    out
}

pub(crate) fn validate(
    op1_bank: u32,
    op2_bank: u32,
    wr_bank: u32,
    op1_col: u32,
    op2_col: u32,
    wr_col: u32,
    iter: u32,
) {
    if iter == 0 {
        panic!("lut: iter must be positive");
    }
    if op1_col != 1 || (op2_col != 1 && op2_col != 4) || wr_col != 1 {
        panic!("lut: requires col=1 input/output and col=1 or col=4 table");
    }
    if op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank {
        panic!("lut: banks must be distinct");
    }
}

fn dst_words(table: &[u8], input: &[u8], lane_table: bool) -> Vec<u128> {
    let mapped: Vec<u8> = input
        .iter()
        .enumerate()
        .map(|(index, &byte)| {
            let channel = index % 16;
            table[(if lane_table { channel * 256 } else { 0 }) + byte as usize]
        })
        .collect();
    pack_bytes(&mapped)
}

// `groups` are the op1/op2/wr base groups; a lane table spans four groups from its base.
fn directed(rob_id: u32, iter: u32, per_lane: bool, groups: [u32; 3]) -> LutCase {
    let nbytes = iter as usize * BYTES_PER_ROW;
    let table = if per_lane {
        lane_table()
    } else {
        shared_table()
    };
    let input = ctest_input(nbytes);
    let src = pack_bytes(&input);
    let lut = if per_lane {
        pack_bytes(&pack_lane_table(&table))
    } else {
        pack_bytes(&table)
    };
    let dst = dst_words(&table, &input, per_lane);
    let op1_bank = 0;
    let op2_bank = 1;
    let wr_bank = 2;
    let op1_col = 1;
    let op2_col = if per_lane { 4 } else { 1 };
    let wr_col = 1;
    validate(op1_bank, op2_bank, wr_bank, op1_col, op2_col, wr_col, iter);
    if src.len() != iter as usize {
        panic!("lut: src rows {} != iter {iter}", src.len());
    }
    let expected_lut_rows = if per_lane { MAX_LUT } else { 16 };
    if lut.len() != expected_lut_rows {
        panic!("lut: table rows {} != {expected_lut_rows}", lut.len());
    }
    if dst.len() != iter as usize {
        panic!("lut: dst rows {} != iter {iter}", dst.len());
    }
    LutCase {
        cmd: LutCmd {
            mode: 0,
            multiplier: 0,
            iter,
            op1_bank,
            op2_bank,
            wr_bank,
            op1_group: groups[0],
            op2_group: groups[1],
            wr_group: groups[2],
            op1_col,
            op2_col,
            wr_col,
            rob_id,
            num_src_words: src.len() as u32,
            num_lut_words: lut.len() as u32,
            num_dst_words: dst.len() as u32,
        },
        src_words: copy_words(&src),
        lut_words: copy_words(&lut),
        dst_words: copy_words(&dst),
    }
}

fn wide(index: u32) -> LutCase {
    let mode = if index == 4 { 1 } else { 2 };
    let rows = match index { 5 => 1, 8 => 3, 9 => 8, _ => 2 };
    let multiplier = match index {
        4 => 0, 5 => 120 << 16, 6 => 4 << 16, 7 => 12 << 16,
        8 => (1 << 30) - 1, 9 => 127 << 16, _ => unreachable!(),
    };
    let input: Vec<i32> = match index {
        4 => vec![i32::MIN, -512, -511, -1, 0, 1, 511, i32::MAX],
        5 => vec![7, 6, 5, 4],
        6 | 7 => vec![0; rows * 4],
        8 => vec![i32::MIN, i32::MAX, -1, 0, -512, 511, 7, 9, 0, 1, 2, 3],
        9 => (0..rows * 4).map(|i| i as i32 % 17 - 8).collect(),
        _ => unreachable!(),
    };
    let table: Vec<i32> = (0..1024).map(|i| {
        if mode == 1 { i * 3 - 100 } else if i <= 30 { 1 << (30 - i) } else { 0 }
    }).collect();
    let source: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
    let table_bytes: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut output = vec![0u8; rows * 16];
    if mode == 1 {
        for (x, word) in input.iter().zip(output.chunks_exact_mut(4)) {
            let at = ((*x).clamp(-512, 511) as u32 & 1023) as usize;
            word.copy_from_slice(&table[at].to_le_bytes());
        }
    } else {
        let peak = i64::from(*input.iter().max().unwrap());
        let values: Vec<u64> = input.iter().map(|x| table[(peak - i64::from(*x)).min(1023) as usize] as u64).collect();
        let sum: u64 = values.iter().sum();
        let reciprocal = (u64::from(multiplier) << 32) / sum;
        for (x, byte) in values.iter().zip(&mut output) {
            let product = x * reciprocal;
            let quotient = product >> 48;
            let remainder = product & ((1 << 48) - 1);
            let rounded = quotient + u64::from(remainder > 1 << 47 || (remainder == 1 << 47 && quotient & 1 != 0));
            *byte = rounded.min(127) as u8;
        }
    }
    LutCase {
        cmd: LutCmd {
            mode, multiplier, iter: rows as u32, op1_bank: 0, op2_bank: 1, wr_bank: 2,
            op1_group: 1, op2_group: 3, wr_group: 2,
            op1_col: 1, op2_col: 1, wr_col: 1, rob_id: index % 4 + 2,
            num_src_words: rows as u32, num_lut_words: 256, num_dst_words: rows as u32,
        },
        src_words: copy_words(&pack_bytes(&source)),
        lut_words: copy_words(&pack_bytes(&table_bytes)),
        dst_words: copy_words(&pack_bytes(&output)),
    }
}
