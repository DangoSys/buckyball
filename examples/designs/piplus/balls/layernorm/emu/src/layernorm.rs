//! LAYERNORM: the layer normalization of FP32 rows, operation for operation
//! in FP32 as workloads/isa/layernorm.h specifies it; the reference the RTL is
//! compared with.
use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use bebop_bemu::ExecContext;

/// Bank rows of one row of length w.
fn row_lines(w: usize) -> usize {
    w.div_ceil(4)
}

/// Lane sums in increasing element order, then (s0 + s1) + (s2 + s3).
fn sum(values: impl Iterator<Item = f32>) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (i, v) in values.enumerate() {
        lanes[i % 4] += v;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// With rms (`beta` None): the mean is 0 and nothing is added.
fn layernorm(x: &[f32], gamma: &[f32], beta: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let inverse = 1.0 / x.len() as f32;
    let mean = if beta.is_some() {
        sum(x.iter().copied()) * inverse
    } else {
        0.0
    };
    let d: Vec<f32> = x.iter().map(|&v| v - mean).collect();
    let r = 1.0 / (sum(d.iter().map(|&v| v * v)) * inverse + eps).sqrt();
    d.iter()
        .enumerate()
        .map(|(i, &v)| match beta {
            Some(beta) => v * r * gamma[i] + beta[i],
            None => v * r * gamma[i],
        })
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let params = rs1_b1(xs1);
    let output = rs1_b2(xs1);
    let rows = rs1_iter(xs1) as usize;
    let w = (xs2 & 0xffff) as usize;
    let in_base = ((xs2 >> 16) & 0x3fff) as usize;
    let out_base = ((xs2 >> 30) & 0x3fff) as usize;
    let param_base = ((xs2 >> 44) & 0x3fff) as usize;
    let rms = (xs2 >> 58) & 1 == 1;
    assert!(
        !rs1_read_old(xs1) && !rs1_update(xs1) && xs2 >> 59 == 0,
        "layernorm: reserved fields must be zero"
    );
    assert!(
        rows > 0 && (16..=2048).contains(&w),
        "layernorm: rows > 0 and 16 <= W <= 2048"
    );
    assert!(
        input != output && params != output && input != params,
        "layernorm: banks must differ"
    );
    for bank in [input, params, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "layernorm: banks must be allocated with one group"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "layernorm: bank rows must be 128 bits");
    let lines = row_lines(w);
    assert!(
        in_base + rows * lines <= bank_lines()
            && out_base + rows * lines <= bank_lines()
            && param_base + if rms { lines + 1 } else { 2 * lines + 1 } <= bank_lines(),
        "layernorm: bank window exceeds capacity"
    );
    let p = &ctx.banks[pbank(ctx, params)];
    let gamma = floats(&p[param_base * 16..param_base * 16 + w * 4]);
    let beta = floats(&p[(param_base + lines) * 16..(param_base + lines) * 16 + w * 4]);
    let eps_at = param_base + if rms { lines } else { 2 * lines };
    let eps = floats(&p[eps_at * 16..eps_at * 16 + 4])[0];
    let source = pbank(ctx, input);
    let mut result = vec![0u8; rows * lines * 16];
    for row in 0..rows {
        let at = (in_base + row * lines) * 16;
        let x = floats(&ctx.banks[source][at..at + w * 4]);
        let y = layernorm(&x, &gamma, (!rms).then_some(&beta[..]), eps);
        assert!(
            x.iter()
                .chain(&gamma)
                .chain(if rms { &[][..] } else { &beta[..] })
                .chain(&y)
                .all(|v| v.is_finite())
                && eps.is_finite(),
            "layernorm: a value not finite"
        );
        for (i, v) in y.iter().enumerate() {
            let at = (row * lines * 4 + i) * 4;
            result[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let destination = pbank(ctx, output);
    ctx.banks[destination][out_base * 16..out_base * 16 + result.len()].copy_from_slice(&result);
    0
}

/// The cycles of the RTL (latency.rs), on the input rows and eps it reads.
pub(crate) fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let w = (xs2 & 0xffff) as usize;
    let param_base = ((xs2 >> 44) & 0x3fff) as usize;
    let rms = (xs2 >> 58) & 1 == 1;
    let lines = row_lines(w);
    let words = |bank: u64| -> Vec<u32> {
        ctx.banks[pbank(ctx, bank)]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let eps = words(rs1_b1(xs1))[(param_base + if rms { lines } else { 2 * lines }) * 4];
    super::latency::cycles(rs1_iter(xs1), xs2, &words(rs1_b0(xs1)), eps)
}
