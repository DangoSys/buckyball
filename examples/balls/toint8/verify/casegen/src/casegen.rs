use crate::model::{quantize_f32, quantize_i32};

pub const KIND_F32: u32 = 0;
pub const KIND_I32: u32 = 1;
pub const BANK_NUM: u32 = 24;
pub const BANK_ENTRIES: u32 = 64;
pub const SCALE_ROWS: usize = 4;
pub const LANES: usize = 4;
pub const MAX_SRC: usize = 16;
pub const MAX_SCALE: usize = 4;
pub const MAX_DST: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToInt8Cmd {
    pub kind: u32,
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
    pub input_base: u32,
    pub num_src_words: u32,
    pub num_scale_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToInt8Case {
    pub cmd: ToInt8Cmd,
    pub src_words: [u128; MAX_SRC],
    pub scale_words: [u128; MAX_SCALE],
    pub dst_words: [u128; MAX_DST],
    pub dst_addr: [u32; MAX_DST],
}

impl ToInt8Case {
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

pub fn gen_case(index: u32, bid: u32) -> ToInt8Case {
    match index {
        0 => f32_signed(bid),
        1 => f32_zero(bid),
        2 => f32_rounding(bid),
        3 => f32_rows(bid),
        4 => i32_scale_rows(bid),
        5 => i32_relu_tile(bid),
        6 => i32_signed(bid),
        7 => i32_offset(bid),
        _ => panic!("toint8: unsupported directed case {index}"),
    }
}

fn pack_u32s(vals: &[u32]) -> Vec<u128> {
    if vals.len() % LANES != 0 {
        panic!("toint8: value count {} is not a multiple of four", vals.len());
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
        panic!("toint8: {} words exceed buffer {N}", src.len());
    }
    let mut out = [0u128; N];
    out[..src.len()].copy_from_slice(src);
    out
}

fn copy_addrs(src: &[u32]) -> [u32; MAX_DST] {
    if src.len() > MAX_DST {
        panic!("toint8: {} dst rows exceed buffer", src.len());
    }
    let mut out = [0u32; MAX_DST];
    out[..src.len()].copy_from_slice(src);
    out
}

fn encode_rs1(op1: u32, op2: u32, wr: u32, iter: u32) -> u64 {
    u64::from(op1) | (u64::from(op2) << 10) | (u64::from(wr) << 20) | (u64::from(iter) << 30)
}

fn encode_i32_rs2(
    relu: bool,
    out_base: u32,
    width: u32,
    height: u32,
    stride: u32,
    in_base: u32,
) -> u64 {
    u64::from(u32::from(relu))
        | (u64::from(out_base) << 1)
        | (u64::from(width) << 8)
        | (u64::from(height) << 15)
        | (u64::from(stride) << 22)
        | (u64::from(in_base) << 29)
}

fn positive_finite(bits: u32) -> bool {
    let value = f32::from_bits(bits);
    value.is_finite() && value > 0.0
}

fn split_rs(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

pub(crate) fn validate(
    kind: u32,
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
    if kind != KIND_F32 && kind != KIND_I32 {
        panic!("toint8: unknown kind {kind}");
    }
    if iter == 0 || iter % 4 != 0 {
        panic!("ToInt8Ball iter must be a positive multiple of four");
    }
    if iter > BANK_ENTRIES {
        panic!("ToInt8Ball input exceeds bank depth");
    }
    if op1_bank >= BANK_NUM {
        panic!("ToInt8Ball input bank is invalid");
    }
    if wr_bank >= BANK_NUM {
        panic!("ToInt8Ball output bank is invalid");
    }
    if op1_col != 1 || wr_col != 1 {
        panic!("ToInt8Ball input and output must each occupy one bank");
    }
    if op1_bank == wr_bank {
        panic!("ToInt8Ball input and output banks must differ");
    }
    if (rs1 & 0x3ff) != u64::from(op1_bank) || ((rs1 >> 20) & 0x3ff) != u64::from(wr_bank) {
        panic!("toint8: rs1 bank fields mismatch");
    }
    if ((rs1 >> 30) & 0x3_ffff_ffff) != u64::from(iter) {
        panic!("toint8: rs1 iter mismatch");
    }
    if kind == KIND_F32 {
        if ((rs1 >> 10) & 0x3ff) != 0 || op2_bank != 0 {
            panic!("QUANT_F32_TO_I8 reserves input bank 1");
        }
        if rs2 >> 32 != 0 {
            panic!("QUANT_F32_TO_I8 reserves rs2[63:32]");
        }
        if !positive_finite(rs2 as u32) {
            panic!("QUANT_F32_TO_I8 scale must be finite and positive");
        }
    } else {
        if op2_bank >= BANK_NUM {
            panic!("QUANT_I32_TO_I8 scale bank is invalid");
        }
        if ((rs1 >> 10) & 0x3ff) != u64::from(op2_bank) {
            panic!("toint8: rs1 scale bank mismatch");
        }
        if op2_col != 1 {
            panic!("QUANT_I32_TO_I8 scale must occupy one bank");
        }
        if op1_bank == op2_bank {
            panic!("QUANT_I32_TO_I8 input and scale banks must differ");
        }
        if op2_bank == wr_bank {
            panic!("QUANT_I32_TO_I8 scale and output banks must differ");
        }
        if rs2 >> 35 != 0 {
            panic!("QUANT_I32_TO_I8 reserves rs2[63:35]");
        }
        let out_base = ((rs2 >> 1) & 0x7f) as u32;
        let width = ((rs2 >> 8) & 0x7f) as u32;
        let height = ((rs2 >> 15) & 0x7f) as u32;
        let stride = ((rs2 >> 22) & 0x7f) as u32;
        let in_base = ((rs2 >> 29) & 0x3f) as u32;
        let output_rows = iter / 4;
        if width == 0 || height == 0 || stride < width {
            panic!("QUANT_I32_TO_I8 output geometry is invalid");
        }
        if width.checked_mul(height) != Some(output_rows) {
            panic!("QUANT_I32_TO_I8 iter does not match output geometry");
        }
        let output_end = out_base
            .checked_add((height - 1).checked_mul(stride).expect("stride overflow"))
            .and_then(|value| value.checked_add(width));
        match output_end {
            Some(end) if end <= BANK_ENTRIES => {}
            _ => panic!("QUANT_I32_TO_I8 output exceeds bank depth"),
        }
        if in_base.checked_add(iter).is_none_or(|end| end > BANK_ENTRIES) {
            panic!("QUANT_I32_TO_I8 input exceeds bank depth");
        }
    }
}

fn f32_dst(src: &[u128], scale: f32) -> Vec<u128> {
    if src.len() % 4 != 0 {
        panic!("QUANT_F32_TO_I8 src rows must be a multiple of four");
    }
    src.chunks(4)
        .map(|group| {
            let mut packed = 0u128;
            for (g, word) in group.iter().enumerate() {
                for lane in 0..LANES {
                    let bits = (*word >> (lane * 32)) as u32;
                    if (bits >> 23) & 0xff == 0xff {
                        panic!("QUANT_F32_TO_I8 input must be finite");
                    }
                    let q = quantize_f32(f32::from_bits(bits), scale, false) as u8;
                    packed |= u128::from(q) << ((g * LANES + lane) * 8);
                }
            }
            packed
        })
        .collect()
}

fn i32_dst(src: &[u128], scales: &[u128], relu: bool) -> Vec<u128> {
    if src.len() % 4 != 0 {
        panic!("QUANT_I32_TO_I8 src rows must be a multiple of four");
    }
    if scales.len() != SCALE_ROWS {
        panic!("QUANT_I32_TO_I8 requires four scale rows");
    }
    let mut scale_vals = [0.0f32; 16];
    for (row, word) in scales.iter().enumerate() {
        for lane in 0..LANES {
            let bits = (*word >> (lane * 32)) as u32;
            if !positive_finite(bits) {
                panic!("QUANT_I32_TO_I8 scales must be finite and positive");
            }
            scale_vals[row * LANES + lane] = f32::from_bits(bits);
        }
    }
    src.chunks(4)
        .map(|group| {
            let mut packed = 0u128;
            for (g, word) in group.iter().enumerate() {
                for lane in 0..LANES {
                    let bits = (*word >> (lane * 32)) as u32;
                    let channel = g * LANES + lane;
                    let q = quantize_i32(bits as i32, scale_vals[channel], relu) as u8;
                    packed |= u128::from(q) << (channel * 8);
                }
            }
            packed
        })
        .collect()
}

fn i32_addrs(iter: u32, out_base: u32, width: u32, stride: u32) -> Vec<u32> {
    let output_rows = iter / 4;
    (0..output_rows)
        .map(|row| out_base + (row / width) * stride + row % width)
        .collect()
}

fn finish(
    kind: u32,
    bid: u32,
    iter: u32,
    op1_bank: u32,
    op2_bank: u32,
    wr_bank: u32,
    op2_col: u32,
    rob_id: u32,
    rs2: u64,
    input_base: u32,
    src: Vec<u128>,
    scale: Vec<u128>,
    dst: Vec<u128>,
    dst_addr: Vec<u32>,
) -> ToInt8Case {
    let op1_col = 1;
    let wr_col = 1;
    let rs1 = encode_rs1(op1_bank, op2_bank, wr_bank, iter);
    validate(
        kind, op1_bank, op2_bank, wr_bank, op1_col, op2_col, wr_col, iter, rs1, rs2,
    );
    if src.len() != iter as usize {
        panic!("toint8: src rows {} != iter {iter}", src.len());
    }
    if dst.len() != (iter / 4) as usize || dst_addr.len() != dst.len() {
        panic!("toint8: dst rows do not match iter/4");
    }
    if kind == KIND_F32 {
        if !scale.is_empty() {
            panic!("QUANT_F32_TO_I8 must not emit scale rows");
        }
    } else if scale.len() != SCALE_ROWS {
        panic!("QUANT_I32_TO_I8 must emit four scale rows");
    }
    let (rs1_lo, rs1_hi) = split_rs(rs1);
    let (rs2_lo, rs2_hi) = split_rs(rs2);
    ToInt8Case {
        cmd: ToInt8Cmd {
            kind,
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
            input_base,
            num_src_words: src.len() as u32,
            num_scale_words: scale.len() as u32,
            num_dst_words: dst.len() as u32,
        },
        src_words: copy_words(&src),
        scale_words: copy_words(&scale),
        dst_words: copy_words(&dst),
        dst_addr: copy_addrs(&dst_addr),
    }
}

fn f32_case(bid: u32, rob_id: u32, scale: f32, values: &[f32]) -> ToInt8Case {
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    let src = pack_u32s(&bits);
    let iter = src.len() as u32;
    let dst = f32_dst(&src, scale);
    let dst_addr: Vec<u32> = (0..dst.len() as u32).collect();
    finish(
        KIND_F32,
        bid,
        iter,
        0,
        0,
        1,
        0,
        rob_id,
        u64::from(scale.to_bits()),
        0,
        src,
        Vec::new(),
        dst,
        dst_addr,
    )
}

fn i32_case(
    bid: u32,
    rob_id: u32,
    op1_bank: u32,
    op2_bank: u32,
    wr_bank: u32,
    relu: bool,
    out_base: u32,
    width: u32,
    height: u32,
    stride: u32,
    in_base: u32,
    values: &[i32],
    scales: &[f32],
) -> ToInt8Case {
    let src = pack_u32s(&values.iter().map(|v| *v as u32).collect::<Vec<_>>());
    let scale = pack_u32s(&scales.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    let iter = src.len() as u32;
    let rs2 = encode_i32_rs2(relu, out_base, width, height, stride, in_base);
    let dst = i32_dst(&src, &scale, relu);
    let dst_addr = i32_addrs(iter, out_base, width, stride);
    finish(
        KIND_I32,
        bid,
        iter,
        op1_bank,
        op2_bank,
        wr_bank,
        1,
        rob_id,
        rs2,
        in_base,
        src,
        scale,
        dst,
        dst_addr,
    )
}

fn f32_signed(bid: u32) -> ToInt8Case {
    f32_case(
        bid,
        2,
        0.5,
        &[
            -300.0, -255.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 254.0, 255.0, 300.0, 2.0, 6.0,
            10.0, 14.0,
        ],
    )
}

fn f32_zero(bid: u32) -> ToInt8Case {
    f32_case(bid, 3, 1.0, &[0.0; 16])
}

fn f32_rounding(bid: u32) -> ToInt8Case {
    f32_case(
        bid,
        4,
        1.0,
        &[
            2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 1.5, -1.5, 63.5, -63.5, 127.5, -128.5, 0.0, 1.0, 2.0,
            3.0,
        ],
    )
}

fn f32_rows(bid: u32) -> ToInt8Case {
    f32_case(
        bid,
        5,
        2.0,
        &[
            0.125, -0.125, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 63.25, 63.75,
            -63.75, -64.75, 0.0, -0.0, 2.25, -2.25, 2.75, -2.75, 3.25, -3.25, 3.75, -3.75, 10.125,
            -10.125, 20.25, -20.25, 0.375, -0.375, 64.25, -65.25,
        ],
    )
}

fn i32_scale_rows(bid: u32) -> ToInt8Case {
    let mut values = [0i32; 16];
    let mut scales = [0.0f32; 16];
    for i in 0..16 {
        values[i] = i as i32 - 8;
        scales[i] = 0.5 + (i as f32) * 0.25;
    }
    i32_case(bid, 6, 0, 1, 2, false, 0, 1, 1, 1, 0, &values, &scales)
}

fn i32_relu_tile(bid: u32) -> ToInt8Case {
    let mut values = [0i32; 64];
    for position in 0..4 {
        for channel in 0..16 {
            values[position * 16 + channel] = position as i32 * 20 + channel as i32 - 8;
        }
    }
    i32_case(bid, 7, 0, 1, 2, true, 3, 2, 2, 4, 0, &values, &[1.0; 16])
}

fn i32_signed(bid: u32) -> ToInt8Case {
    let values: [i32; 16] = [
        -300, -128, -5, -1, 0, 1, 5, 127, 128, 200, -64, 64, -7, 7, 3, -3,
    ];
    i32_case(bid, 8, 0, 1, 2, false, 0, 1, 1, 1, 0, &values, &[1.0; 16])
}

fn i32_offset(bid: u32) -> ToInt8Case {
    let mut values = [0i32; 32];
    for i in 0..32 {
        values[i] = i as i32 - 16;
    }
    i32_case(bid, 9, 0, 1, 2, true, 1, 2, 1, 2, 4, &values, &[0.5; 16])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_cover_both_quant_kinds() {
        let f32 = gen_case(0, 3);
        let i32 = gen_case(5, 3);
        assert_eq!(f32.cmd.kind, KIND_F32);
        assert_eq!(i32.cmd.kind, KIND_I32);
        assert_eq!(f32.cmd.iter % 4, 0);
        assert_eq!(f32.cmd.op1_col, 1);
        assert_eq!(f32.cmd.wr_col, 1);
        assert_eq!(i32.cmd.op2_col, 1);
        assert_eq!(i32.cmd.rs2_lo & 1, 1);
        assert_eq!(i32.cmd.num_scale_words, 4);
        assert_eq!(f32.cmd.num_scale_words, 0);
        assert_eq!(f32.cmd.bid, 3);
    }

    #[test]
    fn former_indices_are_legal() {
        for index in 0..8 {
            let case = gen_case(index, 3);
            assert!(case.cmd.kind == KIND_F32 || case.cmd.kind == KIND_I32);
            assert_eq!(case.cmd.num_src_words, case.cmd.iter);
            assert_eq!(case.cmd.num_dst_words, case.cmd.iter / 4);
        }
    }

    #[test]
    fn f32_ctest_vector_saturates() {
        let case = gen_case(0, 3);
        let packed = case.dst_words[0];
        let bytes: [i8; 16] = [
            -128, -128, -2, -2, 0, 0, 0, 2, 2, 127, 127, 127, 1, 3, 5, 7,
        ];
        for (i, expected) in bytes.iter().enumerate() {
            assert_eq!(((packed >> (i * 8)) as u8) as i8, *expected);
        }
    }

    #[test]
    #[should_panic(expected = "unsupported directed case")]
    fn unknown_index_panics() {
        let _ = gen_case(8, 3);
    }

    #[test]
    #[should_panic(expected = "positive multiple of four")]
    fn illegal_iter_panics() {
        validate(KIND_F32, 0, 0, 1, 1, 0, 1, 3, encode_rs1(0, 0, 1, 3), u64::from(1.0f32.to_bits()));
    }

    #[test]
    #[should_panic(expected = "reserves rs2[63:32]")]
    fn f32_high_rs2_panics() {
        validate(
            KIND_F32,
            0,
            0,
            1,
            1,
            0,
            1,
            4,
            encode_rs1(0, 0, 1, 4),
            u64::from(1.0f32.to_bits()) | (1u64 << 32),
        );
    }

    #[test]
    #[should_panic(expected = "output geometry is invalid")]
    fn i32_zero_width_panics() {
        validate(
            KIND_I32,
            0,
            1,
            2,
            1,
            1,
            1,
            4,
            encode_rs1(0, 1, 2, 4),
            encode_i32_rs2(false, 0, 0, 1, 1, 0),
        );
    }
}
