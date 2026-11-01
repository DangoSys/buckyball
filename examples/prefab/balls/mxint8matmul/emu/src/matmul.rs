//! MXINT8MM: C[M,N] (FP32) = A[M,K] . B[N,K]^T, A and B packed MXINT8 rows.
//! Per C element, in block order b = 0 .. K/32:
//!   p = sum of the 32 INT8 code products of block b (exact INT32)
//!   t = fp32_rne(p * 2^(ea + eb))       ea, eb the block exponents
//!   acc = fp32_rne(acc + t)             acc starts at +0.0, or at the old C
//! Every t and acc must be finite.
use bebop_bemu::thread::banks::{bank_lines, bank_row_bytes};
use bebop_bemu::thread::instructions::decode::{
    pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update,
};
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
    let b_base = ((xs2 >> 34) & 0x3fff) as usize;
    let c_base = ((xs2 >> 48) & 0x3fff) as usize;
    let accumulate = (xs2 >> 62) & 1 == 1;
    assert!(
        xs2 >> 63 == 0 && !rs1_update(xs1),
        "mxint8matmul: reserved fields must be zero"
    );
    assert_eq!(
        rs1_read_old(xs1),
        accumulate,
        "mxint8matmul: rs1[29] must equal the accumulate bit rs2[62]"
    );
    assert!(
        m > 0 && n > 0 && n % 4 == 0,
        "mxint8matmul: M > 0 and N a positive multiple of 4"
    );
    assert!(
        k > 0 && k % 32 == 0,
        "mxint8matmul: K must be a positive multiple of 32"
    );
    assert!(
        a != c && b != c && a != b,
        "mxint8matmul: banks must differ"
    );
    for bank in [a, b, c] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxint8matmul: banks must be allocated with one group"
        );
    }
    assert_eq!(
        bank_row_bytes(),
        16,
        "mxint8matmul: bank rows must be 128 bits"
    );
    let stride = packed_rows(k);
    assert!(
        a_base + m * stride <= bank_lines()
            && b_base + n * stride <= bank_lines()
            && c_base + m * n / 4 <= bank_lines(),
        "mxint8matmul: bank window exceeds capacity"
    );
    let (pa, pb, pc) = (pbank(ctx, a), pbank(ctx, b), pbank(ctx, c));
    let lhs = ctx.banks[pa][a_base * 16..(a_base + m * stride) * 16].to_vec();
    let rhs = ctx.banks[pb][b_base * 16..(b_base + n * stride) * 16].to_vec();
    let window = c_base * 16..(c_base + m * n / 4) * 16;
    let mut result: Vec<f32> = if accumulate {
        ctx.banks[pc][window.clone()]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    } else {
        vec![0.0; m * n]
    };
    assert!(
        result.iter().all(|value| value.is_finite()),
        "mxint8matmul: old C must be finite"
    );
    for (row, lhs) in lhs.chunks_exact(stride * 16).enumerate() {
        for (col, rhs) in rhs.chunks_exact(stride * 16).enumerate() {
            let mut sum = result[row * n + col];
            for block in 0..k / 32 {
                let product: i32 = (0..32)
                    .map(|i| lhs[block * 32 + i] as i8 as i32 * rhs[block * 32 + i] as i8 as i32)
                    .sum();
                let scale = lhs[k + block] as i32 + rhs[k + block] as i32 - 254;
                let term = (product as f64 * f64::from_bits(((scale + 1023) as u64) << 52)) as f32;
                sum += term;
                assert!(
                    term.is_finite() && sum.is_finite(),
                    "mxint8matmul: result overflows FP32"
                );
            }
            result[row * n + col] = sum;
        }
    }
    let bytes: Vec<u8> = result
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    ctx.banks[pc][window].copy_from_slice(&bytes);
    0
}

/// Event cost of tile setup, bank bursts, PE steps, and result acknowledgement.
pub(crate) fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let p = bebop_bemu::configs::ball_domain::parameters::<super::params::Params>(
        "mxint8matmul",
        ctx.ball_id.expect("Ball instance"),
    );
    let tile_m = p.rows.expect("rows") as u64;
    let tile_n = p.cols.expect("cols") as u64;
    let steps = 32 / p.lanes.expect("lanes") as u64;
    let m = xs2 & 0x3ff;
    let n = (xs2 >> 10) & 0x3ff;
    let blocks = rs1_iter(xs1) / 32;
    let mut clocks = 12;
    for row in (0..m).step_by(tile_m as usize) {
        for col in (0..n).step_by(tile_n as usize) {
            let rows = tile_m.min(m - row);
            let cols = tile_n.min(n - col);
            clocks += 2 + blocks * (12 * rows.max(cols) + 8 + steps) + rows * cols;
            if xs2 & (1 << 62) != 0 {
                clocks += rows * cols + 6;
            }
        }
    }
    clocks
}
