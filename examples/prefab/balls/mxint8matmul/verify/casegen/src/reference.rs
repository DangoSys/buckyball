// Reference model of Mxint8matmulBall, ported from the BEMU model
// (emu/src/matmul.rs). MXINT8MM: C[M,N] (FP32) = A[M,K] . B[N,K]^T over
// packed MXINT8 rows of S(K) = K/16 + ceil(K/512) bank rows: the K INT8
// codes, then the K/32 exponent bytes. Per C element, in block order:
//   p   = sum of the 32 code products of the block (exact)
//   t   = fp32_rne(p * 2^(byteA + byteB - 254))
//   acc = fp32_rne(acc + t)     acc starts at +0.0, or at the old C
// C row i holds N little-endian FP32 at bank rows c_base + i*N/4 ..
use std::collections::BTreeMap;

/// Bank rows by (bank, group, row).
pub type Banks = BTreeMap<(u32, u32, u32), u128>;

// Semantic events of one command, reported to the testbench for coverage.
pub const TERM_SUBNORMAL: u32 = 1 << 0;
pub const TERM_ROUNDED: u32 = 1 << 1;
pub const TERM_TIE_DOWN: u32 = 1 << 2;
pub const TERM_TIE_UP: u32 = 1 << 3;
pub const TERM_NEG_ZERO: u32 = 1 << 4;
pub const TERM_LARGE: u32 = 1 << 5;
pub const PRODUCT_EXTREME: u32 = 1 << 6;
pub const SUM_INEXACT: u32 = 1 << 7;
pub const SUM_CANCEL: u32 = 1 << 8;
pub const RESULT_NEG_ZERO: u32 = 1 << 9;
pub const RESULT_SUBNORMAL: u32 = 1 << 10;
pub const CODE_MIN: u32 = 1 << 11;
pub const EXP_ZERO: u32 = 1 << 12;
pub const EXP_HIGH: u32 = 1 << 13;
pub const OLD_SPECIAL: u32 = 1 << 14;

#[derive(Clone, Debug)]
pub struct Command {
    // (bank, group) of A, B and C; the old C is read from (read_bank, C group).
    pub a: (u32, u32),
    pub b: (u32, u32),
    pub c: (u32, u32),
    pub read_bank: u32,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_base: usize,
    pub b_base: usize,
    pub c_base: usize,
    pub accumulate: bool,
}

pub fn stride(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

impl Command {
    pub fn rs2(&self) -> u64 {
        self.m as u64
            | (self.n as u64) << 10
            | (self.a_base as u64) << 20
            | (self.b_base as u64) << 34
            | (self.c_base as u64) << 48
            | u64::from(self.accumulate) << 62
    }
}

fn window(banks: &Banks, bank: u32, group: u32, base: usize, rows: usize) -> Vec<u8> {
    (base..base + rows)
        .flat_map(|row| banks[&(bank, group, row as u32)].to_le_bytes())
        .collect()
}

/// Run one command on `banks`; return its semantic events.
pub fn execute(banks: &mut Banks, entries: usize, cmd: &Command) -> u32 {
    let (m, n, k) = (cmd.m, cmd.n, cmd.k);
    assert!(
        m > 0 && m < 1024 && n > 0 && n < 1024 && n % 4 == 0,
        "mxint8matmul: M > 0 and N a positive multiple of 4"
    );
    assert!(
        k > 0 && k % 32 == 0,
        "mxint8matmul: K must be a positive multiple of 32"
    );
    assert!(
        cmd.a.0 != cmd.b.0
            && cmd.a.0 != cmd.c.0
            && cmd.b.0 != cmd.c.0
            && cmd.a.0 != cmd.read_bank
            && cmd.b.0 != cmd.read_bank,
        "mxint8matmul: banks must differ"
    );
    let s = stride(k);
    assert!(
        cmd.a_base + m * s <= entries && cmd.b_base + n * s <= entries && cmd.c_base + m * n / 4 <= entries,
        "mxint8matmul: bank window exceeds capacity"
    );
    let lhs = window(banks, cmd.a.0, cmd.a.1, cmd.a_base, m * s);
    let rhs = window(banks, cmd.b.0, cmd.b.1, cmd.b_base, n * s);
    let mut features = 0;
    for row in lhs.chunks_exact(s * 16).chain(rhs.chunks_exact(s * 16)) {
        if row[..k].contains(&0x80) {
            features |= CODE_MIN;
        }
        if row[k..k + k / 32].contains(&0) {
            features |= EXP_ZERO;
        }
        if row[k..k + k / 32].iter().any(|&e| e >= 128) {
            features |= EXP_HIGH;
        }
    }
    let mut result: Vec<f32> = if cmd.accumulate {
        window(banks, cmd.read_bank, cmd.c.1, cmd.c_base, m * n / 4)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    } else {
        vec![0.0; m * n]
    };
    assert!(
        result.iter().all(|v| v.is_finite()),
        "mxint8matmul: old C must be finite"
    );
    if cmd.accumulate && result.iter().any(|v| v.to_bits() == 0x8000_0000 || v.is_subnormal()) {
        features |= OLD_SPECIAL;
    }
    for (i, lhs) in lhs.chunks_exact(s * 16).enumerate() {
        for (j, rhs) in rhs.chunks_exact(s * 16).enumerate() {
            let mut sum = result[i * n + j];
            for block in 0..k / 32 {
                let product: i32 = (0..32)
                    .map(|l| lhs[block * 32 + l] as i8 as i32 * rhs[block * 32 + l] as i8 as i32)
                    .sum();
                let scale = lhs[k + block] as i32 + rhs[k + block] as i32 - 254;
                let exact = product as f64 * f64::from_bits(((scale + 1023) as u64) << 52);
                let term = exact as f32;
                let next = sum + term;
                assert!(
                    term.is_finite() && next.is_finite(),
                    "mxint8matmul: result overflows FP32; row={i} col={j} block={block} product={product} scale={scale}"
                );
                if product != 0 {
                    let units = exact.abs() * 2f64.powi(149);
                    if term.is_subnormal() {
                        features |= TERM_SUBNORMAL;
                    }
                    if term as f64 != exact {
                        features |= TERM_ROUNDED;
                    }
                    if exact.abs() < 2f64.powi(-126) && units.fract() == 0.5 {
                        features |= if units.floor() % 2.0 == 0.0 {
                            TERM_TIE_DOWN
                        } else {
                            TERM_TIE_UP
                        };
                    }
                    if term.to_bits() == 0x8000_0000 {
                        features |= TERM_NEG_ZERO;
                    }
                    if exact.abs() >= 2f64.powi(64) {
                        features |= TERM_LARGE;
                    }
                    if product.abs() == 1 << 19 {
                        features |= PRODUCT_EXTREME;
                    }
                }
                // TwoSum: the rounding error of sum + term, exact in FP32.
                let back = next - sum;
                if (sum - (next - back)) + (term - back) != 0.0 {
                    features |= SUM_INEXACT;
                }
                if sum != 0.0 && term != 0.0 && next == 0.0 {
                    features |= SUM_CANCEL;
                }
                sum = next;
            }
            if sum.to_bits() == 0x8000_0000 {
                features |= RESULT_NEG_ZERO;
            }
            if sum.is_subnormal() {
                features |= RESULT_SUBNORMAL;
            }
            result[i * n + j] = sum;
        }
    }
    for (offset, words) in result.chunks_exact(4).enumerate() {
        let mut data = 0u128;
        for (w, value) in words.iter().enumerate() {
            data |= u128::from(value.to_bits()) << (32 * w);
        }
        banks.insert((cmd.c.0, cmd.c.1, (cmd.c_base + offset) as u32), data);
    }
    features
}
