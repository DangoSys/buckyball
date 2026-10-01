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

fn value(code: u8, scale: u8) -> f32 {
    assert!(code & 0x7f != 0x7f && scale != 255, "mxfp8: NaN encoding");
    let exponent = (code >> 3) & 15;
    let mantissa = (code & 7) as f32;
    let element = if exponent == 0 {
        mantissa / 512.0
    } else {
        (8.0 + mantissa) * 2.0f32.powi(exponent as i32 - 10)
    };
    let factor = if scale == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits((scale as u32) << 23)
    };
    let result = element * factor * if code & 128 == 0 { 1.0 } else { -1.0 };
    assert!(result.is_finite(), "mxfp8: decoded value overflows FP32");
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
    let mut av = vec![0f32; rows * k];
    let mut bv = vec![0f32; cols * k];
    for i in 0..av.len() {
        av[i] = value(ctx.banks[pa][i], ctx.banks[pa][rows * k + i / 32]);
    }
    for i in 0..bv.len() {
        bv[i] = value(ctx.banks[pb][i], ctx.banks[pb][cols * k + i / 32]);
    }
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
        for col in 0..cols {
            let mut acc = chain.values[row * cols + col];
            for inner in 0..k {
                acc = av[row * k + inner].mul_add(bv[col * k + inner], acc);
            }
            assert!(acc.is_finite(), "mxfp8: FP32 accumulator overflow");
            chain.values[row * cols + col] = acc;
        }
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
