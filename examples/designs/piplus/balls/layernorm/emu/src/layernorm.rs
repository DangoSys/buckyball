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
    let gamma = p[param_base * 16..param_base * 16 + w * 4].as_chunks::<4>().0;
    let beta = (!rms).then(|| {
        p[(param_base + lines) * 16..(param_base + lines) * 16 + w * 4]
            .as_chunks::<4>()
            .0
    });
    let eps_at = (param_base + if rms { lines } else { 2 * lines }) * 16;
    let eps = f32::from_le_bytes(p[eps_at..eps_at + 4].try_into().unwrap());
    assert!(
        eps.is_finite()
            && gamma
                .iter()
                .chain(beta.unwrap_or(&[]))
                .all(|v| f32::from_le_bytes(*v).is_finite()),
        "layernorm: a value not finite"
    );
    let source = pbank(ctx, input);
    let mut result = vec![0u8; rows * lines * 16];
    let inverse = 1.0 / w as f32;
    for (row, dest) in result.chunks_exact_mut(lines * 16).enumerate() {
        let at = (in_base + row * lines) * 16;
        let x = ctx.banks[source][at..at + w * 4].as_chunks::<4>().0;
        let read = |i: usize| f32::from_le_bytes(x[i]);
        let (mean, shifted) = super::latency::row_stats(w, rms, inverse, eps, read);
        let scale = 1.0 / shifted.sqrt();
        for (i, word) in dest[..w * 4].as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let input = read(i);
            let value = (input - mean) * scale * f32::from_le_bytes(gamma[i]);
            let value = match beta {
                Some(beta) => value + f32::from_le_bytes(beta[i]),
                None => value,
            };
            assert!(input.is_finite() && value.is_finite(), "layernorm: a value not finite");
            *word = value.to_le_bytes();
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
    let input = ctx.banks[pbank(ctx, rs1_b0(xs1))].as_chunks::<4>().0;
    let params = ctx.banks[pbank(ctx, rs1_b1(xs1))].as_chunks::<4>().0;
    let eps = f32::from_le_bytes(params[(param_base + if rms { lines } else { 2 * lines }) * 4]);
    super::latency::cycles_with(rs1_iter(xs1), xs2, |i| f32::from_le_bytes(input[i]), eps)
}
