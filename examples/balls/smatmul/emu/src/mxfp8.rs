use super::super::bank::{bank_lines, bank_row_bytes};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;
use std::cell::RefCell;
use std::collections::HashMap;

struct Chain {
    rows: usize,
    cols: usize,
    bank: u64,
    base: usize,
    values: Vec<f32>,
}

thread_local! {
    static CHAINS: RefCell<HashMap<usize, Chain>> = RefCell::new(HashMap::new());
}

fn value(code: u8, factor: f32) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = (code & 7) as u32;
    let normal = ((exponent as u32 + 120) << 23) | (mantissa << 20);
    let subnormal = (mantissa as f32 / 512.0).to_bits();
    let element = f32::from_bits(if exponent == 0 { subnormal } else { normal });
    let result = f32::from_bits((element * factor).to_bits() | ((code as u32 & 128) << 24));
    result
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let rows = (xs2 & 0xfff) as usize;
    let cols = ((xs2 >> 12) & 0xfff) as usize;
    let k = rs1_iter(xs1) as usize;
    let first = (xs2 >> 24) & 1 != 0;
    let last = (xs2 >> 25) & 1 != 0;
    let base = ((xs2 >> 26) & 63) as usize;
    let a = rs1_b0(xs1);
    let b = rs1_b1(xs1);
    let c = rs1_b2(xs1);
    assert!(
        xs2 >> 32 == 0 && (rows == 1 || rows == 16) && cols == 16 && k > 0 && k % 32 == 0,
        "mxfp8: requires M=1/16, N=16, K multiple of 32 and zero reserved bits"
    );
    assert!(a != b && b != c && a != c, "mxfp8: banks must differ");
    for bank in [a, b, c] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxfp8: banks must be allocated with one column"
        );
    }
    let stride = bank_row_bytes();
    assert!(stride == 16, "mxfp8: requires 16-byte bank rows");
    assert!(
        rows * k + rows * k / 32 <= bank_lines() * stride
            && cols * k + cols * k / 32 <= bank_lines() * stride
            && base * stride + rows * cols * 4 <= bank_lines() * stride,
        "mxfp8: bank footprint exceeds depth"
    );
    let pa = pbank(ctx, a);
    let pb = pbank(ctx, b);
    let a_bytes = &ctx.banks[pa][..rows * k + rows * k / 32];
    let b_bytes = &ctx.banks[pb][..cols * k + cols * k / 32];
    assert!(
        a_bytes[..rows * k]
            .iter()
            .chain(b_bytes[..cols * k].iter())
            .fold(true, |valid, code| valid & (code & 0x7f != 0x7f)),
        "mxfp8: NaN encoding"
    );
    let mut av = vec![0f32; rows * k];
    let mut bv = vec![0f32; cols * k];
    for (block, dst) in av.chunks_exact_mut(32).enumerate() {
        let scale = a_bytes[rows * k + block];
        assert!(scale != 255, "mxfp8: NaN encoding");
        let factor = if scale == 0 {
            f32::from_bits(1 << 22)
        } else {
            f32::from_bits((scale as u32) << 23)
        };
        for (index, decoded) in dst.iter_mut().enumerate() {
            *decoded = value(a_bytes[block * 32 + index], factor);
        }
    }
    for col in 0..cols {
        for block in 0..k / 32 {
            let scale = b_bytes[cols * k + col * (k / 32) + block];
            assert!(scale != 255, "mxfp8: NaN encoding");
            let factor = if scale == 0 {
                f32::from_bits(1 << 22)
            } else {
                f32::from_bits((scale as u32) << 23)
            };
            let codes = &b_bytes[col * k + block * 32..col * k + (block + 1) * 32];
            let mut decoded = [0f32; 32];
            for (decoded, code) in decoded.iter_mut().zip(codes.iter()) {
                *decoded = value(*code, factor);
            }
            for (index, decoded) in decoded.iter().enumerate() {
                bv[(block * 32 + index) * cols + col] = *decoded;
            }
        }
    }
    assert!(
        av.iter()
            .chain(bv.iter())
            .fold(true, |valid, value| valid & value.is_finite()),
        "mxfp8: decoded value overflows FP32"
    );
    let mut chain = CHAINS.with(|chains| {
        let mut chains = chains.borrow_mut();
        if first {
            assert!(
                !chains.contains_key(&ctx.hart_id),
                "mxfp8: first while chain is live"
            );
            Chain {
                rows,
                cols,
                bank: c,
                base,
                values: vec![0f32; rows * cols],
            }
        } else {
            chains
                .remove(&ctx.hart_id)
                .expect("mxfp8: continuation without first")
        }
    });
    assert!(
        chain.rows == rows && chain.cols == cols && chain.bank == c && chain.base == base,
        "mxfp8: continuation changed destination"
    );
    for row in 0..rows {
        if av[row * k..(row + 1) * k].iter().all(|value| *value == 0.0)
            && chain.values[row * cols..(row + 1) * cols]
                .iter()
                .all(|value| value.to_bits() == 0)
        {
            continue;
        }
        let av = &av[row * k..(row + 1) * k];
        let dst = &mut chain.values[row * cols..(row + 1) * cols];
        let mut sums: [f32; 16] = dst.try_into().unwrap();
        for inner in 0..k {
            let a = av[inner];
            let weights = &bv[inner * 16..(inner + 1) * 16];
            for col in 0..16 {
                sums[col] = a.mul_add(weights[col], sums[col]);
            }
        }
        assert!(sums.iter().all(|v| v.is_finite()), "accumulator overflow");
        dst.copy_from_slice(&sums);
    }
    if last {
        let pc = pbank(ctx, c);
        for (i, value) in chain.values.iter().enumerate() {
            ctx.banks[pc][base * stride + i * 4..base * stride + i * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
    } else {
        CHAINS.with(|chains| {
            chains.borrow_mut().insert(ctx.hart_id, chain);
        });
    }
    0
}
