//! MXINT8MM: C[M,N] (FP32) = A[M,K] . B[N,K]^T, A and B packed MXINT8 rows.
//! Per C element, from +0.0 or the old C (accumulate), plus bias[col] (bias,
//! the n values before C's window in the old destination), in block order
//! b = 0 .. K/32:
//!   p = sum of the 32 INT8 code products of block b (exact INT32)
//!   t = fp32_rne(p * 2^(ea + eb))       ea, eb the block exponents
//!   acc = fp32_rne(acc + t)             acc starts at +0.0, or at the old C
//! Every t and acc must be finite.
use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use bebop_bemu::ExecContext;

fn packed_rows(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let (a, b, c) = (rs1_b0(xs1), rs1_b1(xs1), rs1_b2(xs1));
    let k = rs1_iter(xs1) as usize;
    let m = (xs2 & 0x3ff) as usize;
    let n = ((xs2 >> 10) & 0x3ff) as usize;
    let a_base = ((xs2 >> 20) & 0x3fff) as usize;
    let c_pitch = ((xs2 >> 34) & 0x3fff) as usize;
    let c_base = ((xs2 >> 48) & 0x3fff) as usize;
    let accumulate = (xs2 >> 62) & 1 == 1;
    let bias = xs2 >> 63 == 1;
    assert_eq!(
        rs1_read_old(xs1),
        accumulate || bias,
        "mxint8mm: rs1[29] must be set exactly with accumulate rs2[62] or bias rs2[63]"
    );
    assert!(
        m > 0 && n > 0 && n % 4 == 0,
        "mxint8mm: M > 0 and N a positive multiple of 4"
    );
    assert!(k > 0 && k % 32 == 0, "mxint8mm: K must be a positive multiple of 32");
    assert!(a != c && b != c && a != b, "mxint8mm: banks must differ");
    for bank in [a, b, c] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxint8mm: banks must be allocated with one group"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "mxint8mm: bank rows must be 128 bits");
    let stride = packed_rows(k);
    assert!(
        c_pitch >= n / 4
            && a_base + m * stride <= bank_lines()
            && n * stride <= bank_lines()
            && c_base + (m - 1) * c_pitch + n / 4 <= bank_lines()
            && (!bias || c_base >= n / 4),
        "mxint8mm: bank window exceeds capacity"
    );
    let (pa, pb, pc) = (pbank(ctx, a), pbank(ctx, b), pbank(ctx, c));
    let lhs = &ctx.banks[pa][a_base * 16..(a_base + m * stride) * 16];
    let rhs = &ctx.banks[pb][..n * stride * 16];
    // Row r of C: n/4 bank rows from c_base + r * c_pitch.
    let row = |r: usize| (c_base + r * c_pitch) * 16..(c_base + r * c_pitch) * 16 + n * 4;
    let mut result: Vec<f32> = if accumulate {
        (0..m)
            .flat_map(|r| ctx.banks[pc][row(r)].chunks_exact(4))
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    } else {
        vec![0.0; m * n]
    };
    assert!(
        result.iter().all(|value| value.is_finite()),
        "mxint8mm: old C must be finite"
    );
    if bias {
        let row = (c_base - n / 4) * 16;
        let values: Vec<f32> = ctx.banks[pc][row..row + n * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert!(
            values.iter().all(|value| value.is_finite()),
            "mxint8mm: bias must be finite"
        );
        for row in result.chunks_exact_mut(n) {
            for (value, bias) in row.iter_mut().zip(&values) {
                *value += bias;
            }
        }
    }
    bebop_bemu::bank_matrix::mxint8_product(lhs, rhs, stride * 16, k, &mut result, n);
    for (r, values) in result.chunks_exact(n).enumerate() {
        for (dst, value) in ctx.banks[pc][row(r)].chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
    0
}

/// The cycles of the RTL (latency.rs), on the array of this Ball instance.
pub(crate) fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let p =
        bebop_bemu::ball::blink::parameters::<super::params::Params>("mxint8mm", ctx.ball_id.expect("Ball instance"));
    let steps = 32 / p.lanes.expect("lanes") as u64;
    // Only dimensions, accumulate/bias and the PE configuration enter the timing recurrence.
    // Neighbouring tiles commonly use the same shape at different bank addresses.
    type TimingInputs = (u64, u64, u64, u64, u64);
    thread_local! {
        static PREVIOUS: std::cell::Cell<Option<(TimingInputs, u64)>> = const { std::cell::Cell::new(None) };
    }
    let key = (
        rs1_iter(xs1),
        xs2 & (0xfffff | (3 << 62)),
        p.rows.expect("rows") as u64,
        p.cols.expect("cols") as u64,
        steps,
    );
    PREVIOUS.with(|previous| {
        if let Some((shape, cycles)) = previous.get() {
            if shape == key {
                return cycles;
            }
        }
        let cycles = super::latency::cycles(key.0, key.1, key.2, key.3, key.4);
        previous.set(Some((key, cycles)));
        cycles
    })
}
