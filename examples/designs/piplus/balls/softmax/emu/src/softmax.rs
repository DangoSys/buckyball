//! SOFTMAX: the softmax of FP32 rows scaled by a power of two, operation for
//! operation in FP32 as
//! workloads/isa/softmax.h specifies it; the reference RTL is compared with.
use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use bebop_bemu::ExecContext;

const LOG2E: f32 = 1.442_695_040_888_963_4;
/// P(f) ~ 2^f on [0, 1): c1 .. c6.
const COEFFICIENTS: [f32; 6] = [
    0.693_147_180_559_945_3,
    0.240_226_506_959_100_7,
    0.055_504_108_664_821_6,
    0.009_618_129_107_628_5,
    0.001_333_355_814_642_8,
    0.000_154_035_303_933_8,
];

/// Bank rows of one row of length w.
fn row_lines(w: usize) -> usize {
    w.div_ceil(4)
}

/// 2^u of u <= 0, zero below -126.
fn exp2(u: f32) -> f32 {
    if u < -126.0 {
        return 0.0;
    }
    let n = u.floor();
    let f = u - n;
    let mut p = COEFFICIENTS[5];
    for c in COEFFICIENTS[..5].iter().rev() {
        p = p * f + c;
    }
    p = p * f + 1.0;
    // 2^n is normal for n >= -126, and p in [1, 2): the product is exact.
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| exp2((v - m) * LOG2E)).collect();
    let mut lanes = [0.0f32; 4];
    for (i, &v) in e.iter().enumerate() {
        lanes[i % 4] += v;
    }
    let s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let r = 1.0 / s;
    e.iter().map(|&v| v * r).collect()
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    let rows = rs1_iter(xs1) as usize;
    let w = (xs2 & 0xffff) as usize;
    let in_base = ((xs2 >> 16) & 0x3fff) as usize;
    let out_base = ((xs2 >> 30) & 0x3fff) as usize;
    let scale_log2 = ((xs2 >> 44) & 0xff) as u8 as i8 as i32;
    assert!(
        rs1_b1(xs1) == 0 && !rs1_read_old(xs1) && !rs1_update(xs1) && xs2 >> 52 == 0,
        "softmax: reserved fields must be zero"
    );
    assert!(scale_log2 >= -126, "softmax: scale exponent below -126");
    let scale = f32::from_bits(((scale_log2 + 127) as u32) << 23);
    assert!(
        rows > 0 && (16..=2048).contains(&w),
        "softmax: rows > 0 and 16 <= W <= 2048"
    );
    assert!(input != output, "softmax: input and output banks must differ");
    for bank in [input, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "softmax: banks must be allocated with one group"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "softmax: bank rows must be 128 bits");
    let lines = row_lines(w);
    assert!(
        in_base + rows * lines <= bank_lines() && out_base + rows * lines <= bank_lines(),
        "softmax: bank window exceeds capacity"
    );
    let source = pbank(ctx, input);
    let mut result = vec![0u8; rows * lines * 16];
    for row in 0..rows {
        let at = (in_base + row * lines) * 16;
        let x: Vec<f32> = ctx.banks[source][at..at + w * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) * scale)
            .collect();
        assert!(
            x.iter().all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
            "softmax: a value neither finite nor -inf"
        );
        assert!(x.iter().any(|v| v.is_finite()), "softmax: a row without a finite value");
        for (i, y) in softmax(&x).iter().enumerate() {
            let at = (row * lines * 4 + i) * 4;
            result[at..at + 4].copy_from_slice(&y.to_le_bytes());
        }
    }
    let destination = pbank(ctx, output);
    ctx.banks[destination][out_base * 16..out_base * 16 + result.len()].copy_from_slice(&result);
    0
}

/// The cycles of the RTL (latency.rs), on the input rows it reads.
pub(crate) fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let input: Vec<u32> = ctx.banks[pbank(ctx, rs1_b0(xs1))]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    super::latency::cycles(rs1_iter(xs1), xs2, &input)
}
