//! MXINT8MM reference, the contract of the BEMU model (emu/src/matmul.rs):
//! C[M,N] (FP32) = A[M,K] . B[N,K]^T over packed MXINT8 rows (K INT8 codes,
//! then K/32 exponent bytes, zero-padded to whole 16-byte bank rows). Per C
//! element, from +0.0 or the old C (accumulate), plus bias[col] (bias, the n
//! values before C's window in the old destination), in block order:
//!   p = sum of the 32 INT8 code products of the block (exact INT32)
//!   t = fp32_rne(p * 2^(ea + eb - 254))
//!   acc = fp32_rne(acc + t)
//! Every input, t and acc must be finite.

pub const ROW_BYTES: usize = 16;

// Semantic events of a command, sampled by the testbench's covergroup.
pub const F_SUBNORMAL_TERM: u32 = 1 << 0;
pub const F_FLUSHED_TERM: u32 = 1 << 1;
pub const F_HUGE_TERM: u32 = 1 << 2;
pub const F_CANCELLED_BLOCK: u32 = 1 << 3;
pub const F_CODE_MIN: u32 = 1 << 4;
pub const F_EXPONENT_MAX: u32 = 1 << 5;
pub const F_EXPONENT_MIN: u32 = 1 << 6;
pub const F_INEXACT_ADD: u32 = 1 << 7;
pub const F_SPECIAL_START: u32 = 1 << 8;

#[derive(Clone, Copy, Debug)]
pub struct Command {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_base: usize,
    pub c_pitch: usize,
    pub c_base: usize,
    pub accumulate: bool,
    pub bias: bool,
}

impl Command {
    pub fn rs2(&self) -> u64 {
        (self.m as u64)
            | ((self.n as u64) << 10)
            | ((self.a_base as u64) << 20)
            | ((self.c_pitch as u64) << 34)
            | ((self.c_base as u64) << 48)
            | (u64::from(self.accumulate) << 62)
            | (u64::from(self.bias) << 63)
    }

    // The command fits the bank, as MXINT8MM requires.
    pub fn fits(&self, bank_entries: usize) -> bool {
        let stride = packed_rows(self.k);
        self.c_pitch >= self.n / 4
            && self.a_base + self.m * stride <= bank_entries
            && self.n * stride <= bank_entries
            && self.c_base + (self.m - 1) * self.c_pitch + self.n / 4 <= bank_entries
            && (!self.bias || self.c_base >= self.n / 4)
    }
}

pub fn packed_rows(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

fn word(image: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(image[at..at + 4].try_into().unwrap())
}

fn special(value: f32) -> bool {
    value.is_subnormal() || (value == 0.0 && value.is_sign_negative())
}

/// C (row-major m x n) and the semantic events, from the A, B and old
/// destination bank images.
pub fn execute(c: &Command, a: &[u8], b: &[u8], old: &[u8]) -> (Vec<f32>, u32) {
    assert!(
        c.m > 0 && c.n > 0 && c.n % 4 == 0,
        "mxint8mm: M > 0 and N a positive multiple of 4"
    );
    assert!(
        c.k > 0 && c.k % 32 == 0,
        "mxint8mm: K must be a positive multiple of 32"
    );
    assert!(c.fits(a.len() / ROW_BYTES), "mxint8mm: bank window exceeds capacity");
    let (m, n, k) = (c.m, c.n, c.k);
    let stride = packed_rows(k);
    let mut features = 0;
    let mut result = vec![0f32; m * n];
    if c.accumulate {
        for row in 0..m {
            for col in 0..n {
                let value = word(old, (c.c_base + row * c.c_pitch) * ROW_BYTES + col * 4);
                assert!(value.is_finite(), "mxint8mm: old C must be finite");
                if special(value) {
                    features |= F_SPECIAL_START;
                }
                result[row * n + col] = value;
            }
        }
    }
    if c.bias {
        for col in 0..n {
            let value = word(old, (c.c_base - n / 4) * ROW_BYTES + col * 4);
            assert!(value.is_finite(), "mxint8mm: bias must be finite");
            if special(value) {
                features |= F_SPECIAL_START;
            }
            for row in 0..m {
                result[row * n + col] += value;
                assert!(result[row * n + col].is_finite(), "mxint8mm: bias overflows FP32");
            }
        }
    }
    for row in 0..m {
        let lhs = &a[(c.a_base + row * stride) * ROW_BYTES..(c.a_base + (row + 1) * stride) * ROW_BYTES];
        for col in 0..n {
            let rhs = &b[col * stride * ROW_BYTES..(col + 1) * stride * ROW_BYTES];
            let mut sum = result[row * n + col];
            for block in 0..k / 32 {
                let mut any = false;
                let mut product = 0i32;
                for i in block * 32..block * 32 + 32 {
                    let (x, y) = (lhs[i] as i8 as i32, rhs[i] as i8 as i32);
                    if x == -128 || y == -128 {
                        features |= F_CODE_MIN;
                    }
                    any |= x * y != 0;
                    product += x * y;
                }
                let (ea, eb) = (lhs[k + block], rhs[k + block]);
                if ea == 255 || eb == 255 {
                    features |= F_EXPONENT_MAX;
                }
                if ea == 0 || eb == 0 {
                    features |= F_EXPONENT_MIN;
                }
                let scale = ea as i32 + eb as i32 - 254;
                let term = (product as f64 * f64::from_bits(((scale + 1023) as u64) << 52)) as f32;
                if any && product == 0 {
                    features |= F_CANCELLED_BLOCK;
                }
                if term.is_subnormal() {
                    features |= F_SUBNORMAL_TERM;
                }
                if product != 0 && term == 0.0 {
                    features |= F_FLUSHED_TERM;
                }
                if term.abs() >= f32::from_bits(247 << 23) {
                    features |= F_HUGE_TERM;
                }
                let next = sum + term;
                if next as f64 != sum as f64 + term as f64 {
                    features |= F_INEXACT_ADD;
                }
                sum = next;
                assert!(term.is_finite() && sum.is_finite(), "mxint8mm: result overflows FP32");
            }
            result[row * n + col] = sum;
        }
    }
    (result, features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(codes: &[i8], exponents: &[u8]) -> Vec<u8> {
        let mut row: Vec<u8> = codes.iter().map(|&code| code as u8).collect();
        row.extend_from_slice(exponents);
        row.resize(packed_rows(codes.len()) * ROW_BYTES, 0);
        row.resize(1024 * ROW_BYTES, 0);
        row
    }

    fn one(c: i8, d: i8, ea: u8, eb: u8) -> f32 {
        let cmd = Command {
            m: 1,
            n: 4,
            k: 32,
            a_base: 0,
            c_pitch: 1,
            c_base: 0,
            accumulate: false,
            bias: false,
        };
        let mut codes = [0i8; 32];
        codes[0] = c;
        let a = packed(&codes, &[ea]);
        codes[0] = d;
        let mut b = packed(&codes, &[eb]);
        let row = b[..3 * ROW_BYTES].to_vec();
        for col in 1..4 {
            b[col * 3 * ROW_BYTES..(col + 1) * 3 * ROW_BYTES].copy_from_slice(&row);
        }
        execute(&cmd, &a, &b, &[]).0[0]
    }

    #[test]
    fn terms() {
        assert_eq!(one(1, 1, 127, 127), 1.0);
        assert_eq!(one(-128, 127, 127, 127), -16256.0);
        assert_eq!(one(3, 1, 127, 128), 6.0);
        // 2^-149 is the smallest subnormal; 3 * 2^-150 rounds to even 2^-148.
        assert_eq!(one(1, 1, 0, 105), f32::from_bits(1));
        assert_eq!(one(3, 1, 0, 104), f32::from_bits(2));
        assert_eq!(one(1, 1, 0, 103), 0.0);
        assert_eq!(packed_rows(32), 3);
        assert_eq!(packed_rows(512), 33);
        assert_eq!(packed_rows(544), 36);
    }
}
