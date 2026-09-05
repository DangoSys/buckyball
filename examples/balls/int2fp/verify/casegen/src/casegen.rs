use crate::model::int2fp_fp32_bits;

pub const BANK_NUM: u32 = 24;
pub const BANK_ENTRIES: u32 = 64;
pub const LANES: usize = 4;
pub const SCALE_ROWS: usize = 4;
pub const MAX_SRC: usize = 16;
pub const MAX_SCALE: usize = 4;
pub const MAX_DST: usize = 16;
pub const NUM_CASES: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int2FpCmd {
    pub bid: u32,
    pub iter: u32,
    pub relu: u32,
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
    pub num_scale_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int2FpCase {
    pub cmd: Int2FpCmd,
    pub src_words: [u128; MAX_SRC],
    pub scale_words: [u128; MAX_SCALE],
    pub dst_words: [u128; MAX_DST],
}

impl Int2FpCase {
    pub fn src_lo(&self, index: usize) -> u64 {
        self.src_words[index] as u64
    }

    pub fn src_hi(&self, index: usize) -> u64 {
        (self.src_words[index] >> 64) as u64
    }

    pub fn scale_lo(&self, index: usize) -> u64 {
        self.scale_words[index] as u64
    }

    pub fn scale_hi(&self, index: usize) -> u64 {
        (self.scale_words[index] >> 64) as u64
    }

    pub fn dst_lo(&self, index: usize) -> u64 {
        self.dst_words[index] as u64
    }

    pub fn dst_hi(&self, index: usize) -> u64 {
        (self.dst_words[index] >> 64) as u64
    }
}

pub fn gen_case(index: u32, bid: u32) -> Int2FpCase {
    if index >= NUM_CASES {
        panic!("int2fp: unsupported directed case {index}");
    }
    match index {
        0 => directed(bid, 2, 4, false, &MLIR_INPUT, &[0.125; 16]),
        1 => directed(bid, 3, 4, true, &RELU_INPUT, &[1.0; 16]),
        2 => directed(bid, 4, 8, false, &ITER8, &SCALE16),
        3 => directed(bid, 5, 8, true, &ITER8, &SCALE16),
        4 => directed(bid, 6, 16, false, &ITER16, &SCALE16),
        5 => directed(bid, 7, 16, true, &ITER16, &SCALE16),
        _ => panic!("int2fp: unsupported directed case {index}"),
    }
}

const MLIR_INPUT: [i32; 16] = [1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8];
const RELU_INPUT: [i32; 16] = [-8, -1, 0, 1, 2, -3, 4, -5, 6, 7, -8, 9, -10, 11, 12, -13];
const ITER8: [i32; 32] = [
    8, -4, 16, 0, 1, -2, 3, -8, 10, -10, 7, 100, -100, 8, 16, -8, 4, -1, 0, 2, -16, 32, -32, 5, 9,
    -9, 12, -12, 6, -6, 11, -11,
];
const ITER16: [i32; 64] = [
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8, 8, -4, 16, 0, 1, -2, 3, -8, 12, -12,
    6, -6, 9, -9, 11, -11, 20, -20, 24, -24, 15, -15, 18, -18, 30, -30, 40, -40, 25, -25, 28, -28,
    50, -50, 60, -60, 7, -7, 13, -13, 14, -14, 17, -17, 19, -19, 21, -21,
];
const SCALE16: [f32; 16] = [
    0.125, 0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.125, 1.0, 0.125, 0.25, 0.5,
];

fn pack_u32s(vals: &[u32]) -> Vec<u128> {
    if vals.len() % LANES != 0 {
        panic!("int2fp: value count {} is not a multiple of four", vals.len());
    }
    vals.chunks(LANES)
        .map(|chunk| {
            let mut word = 0u128;
            for (lane, value) in chunk.iter().enumerate() {
                word |= u128::from(*value) << (lane * 32);
            }
            word
        })
        .collect()
}

fn copy_words<const N: usize>(src: &[u128]) -> [u128; N] {
    if src.len() > N {
        panic!("int2fp: {} words exceed buffer {N}", src.len());
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

fn positive_finite(bits: u32) -> bool {
    bits >> 31 == 0 && ((bits >> 23) & 0xff) != 0xff && (bits & 0x7fff_ffff) != 0
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
    scales: &[u128],
) {
    if iter == 0 || iter % 4 != 0 {
        panic!("INT32_TO_FP32 iter must be a positive multiple of four");
    }
    if iter > BANK_ENTRIES {
        panic!("INT32_TO_FP32 input exceeds bank depth");
    }
    if op1_bank >= BANK_NUM || op2_bank >= BANK_NUM || wr_bank >= BANK_NUM {
        panic!("INT32_TO_FP32 bank id is invalid");
    }
    if op1_col != 1 || op2_col != 1 || wr_col != 1 {
        panic!("INT32_TO_FP32 operands must each occupy one bank");
    }
    if op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank {
        panic!("INT32_TO_FP32 banks must be distinct");
    }
    if (rs1 & 0x3ff) != u64::from(op1_bank)
        || ((rs1 >> 10) & 0x3ff) != u64::from(op2_bank)
        || ((rs1 >> 20) & 0x3ff) != u64::from(wr_bank)
    {
        panic!("int2fp: rs1 bank fields mismatch");
    }
    if ((rs1 >> 30) & 0x3_ffff_ffff) != u64::from(iter) {
        panic!("int2fp: rs1 iter mismatch");
    }
    if rs2 >> 1 != 0 {
        panic!("INT32_TO_FP32 reserves rs2[63:1]");
    }
    if scales.len() != SCALE_ROWS {
        panic!("INT32_TO_FP32 must emit four scale rows");
    }
    for word in scales {
        for lane in 0..LANES {
            let bits = (*word >> (lane * 32)) as u32;
            if !positive_finite(bits) {
                panic!("INT32_TO_FP32 scales must be finite and positive");
            }
        }
    }
}

fn dst_words(src: &[u128], scales: &[u128], relu: bool) -> Vec<u128> {
    src.iter()
        .enumerate()
        .map(|(row, word)| {
            let scale = scales[row % SCALE_ROWS];
            let mut packed = 0u128;
            for lane in 0..LANES {
                let raw = (*word >> (lane * 32)) as u32 as i32;
                let value = if relu { raw.max(0) } else { raw };
                let bits = int2fp_fp32_bits(value, (scale >> (lane * 32)) as u32);
                packed |= u128::from(bits) << (lane * 32);
            }
            packed
        })
        .collect()
}

fn directed(bid: u32, rob_id: u32, iter: u32, relu: bool, values: &[i32], scales: &[f32]) -> Int2FpCase {
    if values.len() != iter as usize * LANES {
        panic!(
            "int2fp: {} values do not match iter {iter} * {LANES}",
            values.len()
        );
    }
    if scales.len() != SCALE_ROWS * LANES {
        panic!("int2fp: {} scales, expected {}", scales.len(), SCALE_ROWS * LANES);
    }
    let src = pack_u32s(&values.iter().map(|v| *v as u32).collect::<Vec<_>>());
    let scale = pack_u32s(&scales.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    let op1_bank = 0;
    let op2_bank = 1;
    let wr_bank = 2;
    let op1_col = 1;
    let op2_col = 1;
    let wr_col = 1;
    let rs1 = encode_rs1(op1_bank, op2_bank, wr_bank, iter);
    let rs2 = u64::from(relu);
    validate(
        op1_bank, op2_bank, wr_bank, op1_col, op2_col, wr_col, iter, rs1, rs2, &scale,
    );
    if src.len() != iter as usize {
        panic!("int2fp: src rows {} != iter {iter}", src.len());
    }
    let dst = dst_words(&src, &scale, relu);
    if dst.len() != iter as usize {
        panic!("int2fp: dst rows {} != iter {iter}", dst.len());
    }
    let (rs1_lo, rs1_hi) = split_rs(rs1);
    let (rs2_lo, rs2_hi) = split_rs(rs2);
    Int2FpCase {
        cmd: Int2FpCmd {
            bid,
            iter,
            relu: u32::from(relu),
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
            num_scale_words: scale.len() as u32,
            num_dst_words: dst.len() as u32,
        },
        src_words: copy_words(&src),
        scale_words: copy_words(&scale),
        dst_words: copy_words(&dst),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_hit_relu_and_iter_bins() {
        let mut relu = [false; 2];
        let mut iter = [false; 3];
        for index in 0..NUM_CASES {
            let case = gen_case(index, 4);
            assert_eq!(case.cmd.bid, 4);
            assert_eq!(case.cmd.op1_col, 1);
            assert_eq!(case.cmd.op2_col, 1);
            assert_eq!(case.cmd.wr_col, 1);
            assert_eq!(case.cmd.num_src_words, case.cmd.iter);
            assert_eq!(case.cmd.num_dst_words, case.cmd.iter);
            assert_eq!(case.cmd.num_scale_words, SCALE_ROWS as u32);
            assert_eq!(case.cmd.rs2_lo & 1, case.cmd.relu);
            assert_eq!(case.cmd.rs2_lo >> 1, 0);
            assert_eq!(case.cmd.rs2_hi, 0);
            relu[case.cmd.relu as usize] = true;
            match case.cmd.iter {
                4 => iter[0] = true,
                8 => iter[1] = true,
                16 => iter[2] = true,
                other => panic!("unexpected iter {other}"),
            }
        }
        assert!(relu.iter().all(|hit| *hit));
        assert!(iter.iter().all(|hit| *hit));
    }

    #[test]
    fn case_zero_matches_mlir_scale() {
        let case = gen_case(0, 4);
        assert_eq!(case.cmd.iter, 4);
        assert_eq!(case.cmd.relu, 0);
        assert_eq!(case.dst_words[0] as u32, 0.125f32.to_bits());
        assert_eq!((case.dst_words[0] >> 32) as u32, 0.25f32.to_bits());
        assert_eq!((case.dst_words[0] >> 64) as u32, 0.375f32.to_bits());
        assert_eq!((case.dst_words[0] >> 96) as u32, (-0.125f32).to_bits());
    }

    #[test]
    fn case_one_relu_clamps_negatives() {
        let case = gen_case(1, 4);
        assert_eq!(case.cmd.iter, 4);
        assert_eq!(case.cmd.relu, 1);
        assert_eq!(case.dst_words[0] as u32, 0);
        assert_eq!((case.dst_words[0] >> 32) as u32, 0);
        assert_eq!((case.dst_words[0] >> 64) as u32, 0);
        assert_eq!((case.dst_words[0] >> 96) as u32, 1.0f32.to_bits());
    }

    #[test]
    #[should_panic(expected = "unsupported directed case")]
    fn unknown_index_panics() {
        let _ = gen_case(NUM_CASES, 4);
    }

    #[test]
    #[should_panic(expected = "positive multiple of four")]
    fn illegal_iter_panics() {
        validate(0, 1, 2, 1, 1, 1, 3, encode_rs1(0, 1, 2, 3), 0, &[0; 4]);
    }

    #[test]
    #[should_panic(expected = "banks must be distinct")]
    fn overlapping_banks_panics() {
        validate(0, 0, 2, 1, 1, 1, 4, encode_rs1(0, 0, 2, 4), 0, &[0; 4]);
    }

    #[test]
    #[should_panic(expected = "reserves rs2[63:1]")]
    fn reserved_rs2_panics() {
        let scale = pack_u32s(&[1.0f32.to_bits(); 16]);
        validate(0, 1, 2, 1, 1, 1, 4, encode_rs1(0, 1, 2, 4), 2, &scale);
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn nonpositive_scale_panics() {
        let scale = pack_u32s(&[0u32; 16]);
        validate(0, 1, 2, 1, 1, 1, 4, encode_rs1(0, 1, 2, 4), 0, &scale);
    }
}
