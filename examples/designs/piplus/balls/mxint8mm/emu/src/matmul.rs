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
    let lhs = ctx.banks[pa][a_base * 16..(a_base + m * stride) * 16].to_vec();
    let rhs = ctx.banks[pb][..n * stride * 16].to_vec();
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
        for (index, value) in result.iter_mut().enumerate() {
            *value += values[index % n];
        }
    }
    for (row, lhs) in lhs.chunks_exact(stride * 16).enumerate() {
        for (col, rhs) in rhs.chunks_exact(stride * 16).enumerate() {
            let mut sum = result[row * n + col];
            // The codes of a row, then the exponents of its blocks.
            let (lhs_codes, lhs_exponents) = lhs.split_at(k);
            let (rhs_codes, rhs_exponents) = rhs.split_at(k);
            for (block, (x, y)) in lhs_codes.chunks_exact(32).zip(rhs_codes.chunks_exact(32)).enumerate() {
                let product: i32 = x.iter().zip(y).map(|(&x, &y)| x as i8 as i32 * y as i8 as i32).sum();
                let scale = lhs_exponents[block] as i32 + rhs_exponents[block] as i32 - 254;
                let term = (product as f64 * f64::from_bits(((scale + 1023) as u64) << 52)) as f32;
                sum += term;
                assert!(term.is_finite() && sum.is_finite(), "mxint8mm: result overflows FP32");
            }
            result[row * n + col] = sum;
        }
    }
    for (r, values) in result.chunks_exact(n).enumerate() {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        ctx.banks[pc][row(r)].copy_from_slice(&bytes);
    }
    0
}

/// The cycles of the RTL (latency.rs), on the array of this Ball instance.
pub(crate) fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let p =
        bebop_bemu::ball::blink::parameters::<super::params::Params>("mxint8mm", ctx.ball_id.expect("Ball instance"));
    let steps = 32 / p.lanes.expect("lanes") as u64;
    super::latency::cycles(
        rs1_iter(xs1),
        xs2,
        p.rows.expect("rows") as u64,
        p.cols.expect("cols") as u64,
        steps,
    )
}
