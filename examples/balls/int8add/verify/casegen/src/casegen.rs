pub const ROW_BYTES: usize = 16;
pub const BANK_ENTRIES: u32 = 64;
pub const MAX_ROWS: usize = 64;
pub const NUM_CASES: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8AddCmd {
    pub relu: u32,
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
    pub num_lhs_words: u32,
    pub num_rhs_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int8AddCase {
    pub cmd: Int8AddCmd,
    pub lhs_words: [u128; MAX_ROWS],
    pub rhs_words: [u128; MAX_ROWS],
    pub dst_words: [u128; MAX_ROWS],
}

impl Int8AddCase {
    pub fn lhs_lo(&self, index: usize) -> u64 {
        self.lhs_words[index] as u64
    }

    pub fn lhs_hi(&self, index: usize) -> u64 {
        (self.lhs_words[index] >> 64) as u64
    }

    pub fn rhs_lo(&self, index: usize) -> u64 {
        self.rhs_words[index] as u64
    }

    pub fn rhs_hi(&self, index: usize) -> u64 {
        (self.rhs_words[index] >> 64) as u64
    }

    pub fn dst_lo(&self, index: usize) -> u64 {
        self.dst_words[index] as u64
    }

    pub fn dst_hi(&self, index: usize) -> u64 {
        (self.dst_words[index] >> 64) as u64
    }
}

pub fn gen_case(index: u32, bid: u32) -> Int8AddCase {
    if index >= NUM_CASES {
        panic!("int8add: unsupported directed case {index}");
    }
    match index {
        0 => directed(bid, 2, false),
        1 => directed(bid, 3, true),
        _ => panic!("int8add: unsupported directed case {index}"),
    }
}

fn compute(lhs: i8, rhs: i8, lhs_ratio: f32, rhs_ratio: f32, relu: bool) -> i8 {
    let value = (lhs as f32) * lhs_ratio + (rhs as f32) * rhs_ratio;
    let rounded = value.round_ties_even();
    let clamped = if relu {
        rounded.max(0.0)
    } else {
        rounded
    }
    .clamp(-128.0, 127.0);
    clamped as i8
}

fn positive_finite(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn ctest_values(rows: usize) -> (Vec<u128>, Vec<u128>) {
    let count = rows * ROW_BYTES;
    let mut lhs = vec![0u8; count];
    let mut rhs = vec![0u8; count];
    for i in 0..count {
        lhs[i] = ((i as i32 * 37 & 255) - 128) as i8 as u8;
        rhs[i] = ((i as i32 * 19 & 255) - 128) as i8 as u8;
    }
    (pack_bytes(&lhs), pack_bytes(&rhs))
}

fn pack_bytes(bytes: &[u8]) -> Vec<u128> {
    if bytes.len() % ROW_BYTES != 0 {
        panic!(
            "int8add: {} bytes is not a multiple of {ROW_BYTES}",
            bytes.len()
        );
    }
    bytes
        .chunks(ROW_BYTES)
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
        panic!("int8add: {} words exceed buffer {N}", src.len());
    }
    let mut out = [0u128; N];
    out[..src.len()].copy_from_slice(src);
    out
}

fn encode_rs1(op1: u32, op2: u32, wr: u32, iter: u32) -> u64 {
    u64::from(op1) | (u64::from(op2) << 10) | (u64::from(wr) << 20) | (u64::from(iter) << 30)
}

fn encode_rs2(lhs_ratio: f32, rhs_ratio: f32) -> u64 {
    u64::from(lhs_ratio.to_bits()) | (u64::from(rhs_ratio.to_bits()) << 32)
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
    lhs_ratio: f32,
    rhs_ratio: f32,
) {
    if iter == 0 || iter > BANK_ENTRIES {
        panic!("int8add: iter must fit in one physical bank");
    }
    if op1_col != 1 || op2_col != 1 || wr_col != 1 {
        panic!("int8add: bank groups must match");
    }
    if op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank {
        panic!("int8add: banks must be distinct");
    }
    if !positive_finite(lhs_ratio) || !positive_finite(rhs_ratio) {
        panic!("int8add: ratios must be finite and positive");
    }
    if (rs1 & 0x3ff) != u64::from(op1_bank)
        || ((rs1 >> 10) & 0x3ff) != u64::from(op2_bank)
        || ((rs1 >> 20) & 0x3ff) != u64::from(wr_bank)
    {
        panic!("int8add: rs1 bank fields mismatch");
    }
    if ((rs1 >> 30) & 0x3_ffff_ffff) != u64::from(iter) {
        panic!("int8add: rs1 iter mismatch");
    }
    if rs2 != encode_rs2(lhs_ratio, rhs_ratio) {
        panic!("int8add: rs2 ratio fields mismatch");
    }
}

fn dst_words(
    lhs: &[u128],
    rhs: &[u128],
    iter: u32,
    lhs_ratio: f32,
    rhs_ratio: f32,
    relu: bool,
) -> Vec<u128> {
    let mut out = Vec::with_capacity(iter as usize);
    for row in 0..iter as usize {
        let mut packed = 0u128;
        for lane in 0..ROW_BYTES {
            let l = ((lhs[row] >> (lane * 8)) & 0xff) as u8 as i8;
            let r = ((rhs[row] >> (lane * 8)) & 0xff) as u8 as i8;
            let q = compute(l, r, lhs_ratio, rhs_ratio, relu);
            packed |= u128::from(q as u8) << (lane * 8);
        }
        out.push(packed);
    }
    out
}

fn directed(bid: u32, rob_id: u32, relu: bool) -> Int8AddCase {
    let iter = 7u32;
    let lhs_ratio = 0.5f32;
    let rhs_ratio = 0.25f32;
    let (lhs, rhs) = ctest_values(iter as usize);
    let op1_bank = 0;
    let op2_bank = 1;
    let wr_bank = 2;
    let op1_col = 1;
    let op2_col = 1;
    let wr_col = 1;
    let rs1 = encode_rs1(op1_bank, op2_bank, wr_bank, iter);
    let rs2 = encode_rs2(lhs_ratio, rhs_ratio);
    validate(
        op1_bank,
        op2_bank,
        wr_bank,
        op1_col,
        op2_col,
        wr_col,
        iter,
        rs1,
        rs2,
        lhs_ratio,
        rhs_ratio,
    );
    if lhs.len() != iter as usize || rhs.len() != iter as usize {
        panic!("int8add: operand rows != iter {iter}");
    }
    let dst = dst_words(&lhs, &rhs, iter, lhs_ratio, rhs_ratio, relu);
    let (rs1_lo, rs1_hi) = split_rs(rs1);
    let (rs2_lo, rs2_hi) = split_rs(rs2);
    Int8AddCase {
        cmd: Int8AddCmd {
            relu: u32::from(relu),
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
            num_lhs_words: lhs.len() as u32,
            num_rhs_words: rhs.len() as u32,
            num_dst_words: dst.len() as u32,
        },
        lhs_words: copy_words(&lhs),
        rhs_words: copy_words(&rhs),
        dst_words: copy_words(&dst),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctest_expected(a: i8, b: i8, relu: bool) -> i8 {
        compute(a, b, 0.5, 0.25, relu)
    }

    #[test]
    fn case_zero_matches_ctest() {
        let case = gen_case(0, 7);
        assert_eq!(case.cmd.bid, 7);
        assert_eq!(case.cmd.relu, 0);
        assert_eq!(case.cmd.iter, 7);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.op2_bank, 1);
        assert_eq!(case.cmd.wr_bank, 2);
        assert_eq!(case.cmd.num_lhs_words, 7);
        assert_eq!(case.cmd.num_rhs_words, 7);
        assert_eq!(case.cmd.num_dst_words, 7);
        let lhs0 = ((case.lhs_words[0] >> 0) & 0xff) as u8 as i8;
        let rhs0 = ((case.rhs_words[0] >> 0) & 0xff) as u8 as i8;
        assert_eq!(lhs0, -128);
        assert_eq!(rhs0, -128);
        let got0 = (case.dst_words[0] & 0xff) as u8 as i8;
        assert_eq!(got0, ctest_expected(lhs0, rhs0, false));
    }

    #[test]
    fn cases_hit_funct7_bins() {
        let mut hit = [false; 2];
        for index in 0..NUM_CASES {
            let case = gen_case(index, 7);
            assert_eq!(case.cmd.bid, 7);
            assert_eq!(case.cmd.op1_col, 1);
            assert_eq!(case.cmd.op2_col, 1);
            assert_eq!(case.cmd.wr_col, 1);
            assert_eq!(case.cmd.num_lhs_words, case.cmd.iter);
            assert_eq!(case.cmd.num_rhs_words, case.cmd.iter);
            assert_eq!(case.cmd.num_dst_words, case.cmd.iter);
            hit[case.cmd.relu as usize] = true;
        }
        assert!(hit.iter().all(|h| *h));
    }

    #[test]
    fn relu_floors_negatives() {
        let case = gen_case(1, 7);
        assert_eq!(case.cmd.relu, 1);
        for row in 0..case.cmd.iter as usize {
            for lane in 0..ROW_BYTES {
                let got = ((case.dst_words[row] >> (lane * 8)) & 0xff) as u8 as i8;
                assert!(got >= 0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "unsupported directed case")]
    fn unknown_index_panics() {
        let _ = gen_case(NUM_CASES, 7);
    }

    #[test]
    #[should_panic(expected = "iter must fit")]
    fn illegal_iter_panics() {
        validate(0, 1, 2, 1, 1, 1, 0, encode_rs1(0, 1, 2, 0), encode_rs2(0.5, 0.25), 0.5, 0.25);
    }

    #[test]
    #[should_panic(expected = "banks must be distinct")]
    fn overlapping_banks_panics() {
        validate(0, 0, 2, 1, 1, 1, 7, encode_rs1(0, 0, 2, 7), encode_rs2(0.5, 0.25), 0.5, 0.25);
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn nonpositive_ratio_panics() {
        validate(0, 1, 2, 1, 1, 1, 7, encode_rs1(0, 1, 2, 7), encode_rs2(0.0, 0.25), 0.0, 0.25);
    }
}
