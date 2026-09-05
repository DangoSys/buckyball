pub const BYTES_PER_ROW: usize = 16;
pub const SHARED_TABLE_BYTES: usize = 256;
pub const LANE_TABLE_BYTES: usize = 16 * SHARED_TABLE_BYTES;
pub const MAX_SRC: usize = 8;
pub const MAX_LUT: usize = LANE_TABLE_BYTES / BYTES_PER_ROW;
pub const MAX_DST: usize = 8;
pub const NUM_CASES: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LutCmd {
    pub bid: u32,
    pub iter: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub op1_col: u32,
    pub op2_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub rs1_lo: u32,
    pub rs1_hi: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
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

pub fn gen_case(index: u32, bid: u32) -> LutCase {
    if index >= NUM_CASES {
        panic!("lut: unsupported directed case {index}");
    }
    match index {
        0 => directed(bid, 2, 4, false),
        1 => directed(bid, 3, 1, false),
        2 => directed(bid, 4, 8, false),
        3 => directed(bid, 5, 4, true),
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
                packed[row * 64 + group * 16 + byte] =
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

fn encode_rs1(op1: u32, op2: u32, wr: u32, iter: u32) -> u64 {
    u64::from(op1) | (u64::from(op2) << 10) | (u64::from(wr) << 20) | (u64::from(iter) << 30)
}

fn split_rs(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

pub(crate) fn validate(
    op1_bank: u32,
    op2_bank: u32,
    wr_bank: u32,
    op1_col: u32,
    op2_col: u32,
    wr_col: u32,
    iter: u32,
    rs1: u64,
    rs2: u64,
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
    if (rs1 & 0x3ff) != u64::from(op1_bank)
        || ((rs1 >> 10) & 0x3ff) != u64::from(op2_bank)
        || ((rs1 >> 20) & 0x3ff) != u64::from(wr_bank)
    {
        panic!("lut: rs1 bank fields mismatch");
    }
    if ((rs1 >> 30) & 0x3_ffff_ffff) != u64::from(iter) {
        panic!("lut: rs1 iter mismatch");
    }
    if rs2 != 0 {
        panic!("lut: rs2 must be zero");
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

fn directed(bid: u32, rob_id: u32, iter: u32, per_lane: bool) -> LutCase {
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
    let rs1 = encode_rs1(op1_bank, op2_bank, wr_bank, iter);
    let rs2 = 0u64;
    validate(
        op1_bank, op2_bank, wr_bank, op1_col, op2_col, wr_col, iter, rs1, rs2,
    );
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
    let (rs1_lo, rs1_hi) = split_rs(rs1);
    let (rs2_lo, rs2_hi) = split_rs(rs2);
    LutCase {
        cmd: LutCmd {
            bid,
            iter,
            op1_bank,
            op2_bank,
            wr_bank,
            op1_col,
            op2_col,
            wr_col,
            rob_id,
            rs1_lo,
            rs1_hi,
            rs2_lo,
            rs2_hi,
            num_src_words: src.len() as u32,
            num_lut_words: lut.len() as u32,
            num_dst_words: dst.len() as u32,
        },
        src_words: copy_words(&src),
        lut_words: copy_words(&lut),
        dst_words: copy_words(&dst),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_zero_matches_ctest() {
        let case = gen_case(0, 5);
        assert_eq!(case.cmd.iter, 4);
        assert_eq!(case.cmd.bid, 5);
        assert_eq!(case.cmd.num_src_words, 4);
        assert_eq!(case.cmd.num_lut_words, 16);
        assert_eq!(case.cmd.num_dst_words, 4);
        let table = shared_table();
        let input = ctest_input(64);
        assert_eq!(table[0], ((19 ^ 0xa5) & 0xff) as u8);
        assert_eq!(input[0], (-128i8) as u8);
        let exp0 = table[input[0] as usize];
        assert_eq!(case.dst_words[0] as u8, exp0);
        for i in 0..16 {
            let got = ((case.src_words[0] >> (i * 8)) & 0xff) as u8;
            assert_eq!(got, input[i]);
        }
        for i in 0..16 {
            let got = ((case.lut_words[0] >> (i * 8)) & 0xff) as u8;
            assert_eq!(got, table[i]);
        }
    }

    #[test]
    fn cases_hit_iter_bins() {
        let mut hit = [false; 3];
        for index in 0..NUM_CASES {
            let case = gen_case(index, 5);
            assert_eq!(case.cmd.bid, 5);
            assert_eq!(case.cmd.op1_col, 1);
            assert!(case.cmd.op2_col == 1 || case.cmd.op2_col == 4);
            assert_eq!(case.cmd.wr_col, 1);
            assert_eq!(case.cmd.rs2_lo, 0);
            assert_eq!(case.cmd.rs2_hi, 0);
            assert_eq!(case.cmd.num_src_words, case.cmd.iter);
            assert_eq!(case.cmd.num_dst_words, case.cmd.iter);
            assert_eq!(
                case.cmd.num_lut_words,
                if case.cmd.op2_col == 4 { 256 } else { 16 }
            );
            match case.cmd.iter {
                1 => hit[0] = true,
                4 => hit[1] = true,
                8 => hit[2] = true,
                other => panic!("unexpected iter {other}"),
            }
        }
        assert!(hit.iter().all(|h| *h));
    }

    #[test]
    fn lane_case_selects_channel_table() {
        let case = gen_case(3, 5);
        let input = ctest_input(64);
        let table = lane_table();
        assert_eq!(case.cmd.op2_col, 4);
        assert_eq!(case.cmd.num_lut_words, 256);
        for lane in 0..16 {
            let got = ((case.dst_words[0] >> (lane * 8)) & 0xff) as u8;
            assert_eq!(got, table[lane * 256 + input[lane] as usize]);
        }
    }

    #[test]
    #[should_panic(expected = "unsupported directed case")]
    fn unknown_index_panics() {
        let _ = gen_case(NUM_CASES, 5);
    }

    #[test]
    #[should_panic(expected = "iter must be positive")]
    fn illegal_iter_panics() {
        validate(0, 1, 2, 1, 1, 1, 0, encode_rs1(0, 1, 2, 0), 0);
    }

    #[test]
    #[should_panic(expected = "banks must be distinct")]
    fn overlapping_banks_panics() {
        validate(0, 0, 2, 1, 1, 1, 4, encode_rs1(0, 0, 2, 4), 0);
    }

    #[test]
    #[should_panic(expected = "rs2 must be zero")]
    fn reserved_rs2_panics() {
        validate(0, 1, 2, 1, 1, 1, 4, encode_rs1(0, 1, 2, 4), 1);
    }
}
