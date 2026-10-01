use super::super::bank::{bank_lines, bank_row_bytes};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;
use std::{cell::RefCell, collections::HashMap};

struct Chain {
    rows: usize,
    bank: u64,
    base: usize,
    values: Vec<f32>,
}

thread_local! {
    static CHAINS: RefCell<HashMap<usize, Chain>> = RefCell::new(HashMap::new());
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
    assert!(xs2 >> 32 == 0 && (rows == 1 || rows == 16) && cols == 16 && k > 0 && k % 4 == 0,
            "f32 matmul: requires M=1/16, N=16, positive K multiple of four and zero reserved bits");
    assert!(a != b && b != c && a != c, "f32 matmul: banks must differ");
    for bank in [a, b, c] {
        assert!(ctx.config(bank).allocated && ctx.config(bank).cols == 1,
                "f32 matmul: banks must be allocated with one column");
    }
    let stride = bank_row_bytes();
    assert!(stride == 16 && rows * k * 4 <= bank_lines() * stride && cols * k * 4 <= bank_lines() * stride
            && base * stride + rows * cols * 4 <= bank_lines() * stride, "f32 matmul: bank footprint exceeds depth");
    let pa = pbank(ctx, a);
    let pb = pbank(ctx, b);
    let av: Vec<_> = ctx.banks[pa][..rows * k * 4].chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect();
    let bv: Vec<_> = ctx.banks[pb][..cols * k * 4].chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect();
    assert!(av.iter().chain(&bv).all(|value| value.is_finite()), "f32 matmul: non-finite operand");
    let mut chain = CHAINS.with(|chains| {
        let mut chains = chains.borrow_mut();
        if first {
            assert!(!chains.contains_key(&ctx.hart_id), "f32 matmul: first while chain is live");
            Chain { rows, bank: c, base, values: vec![0f32; rows * cols] }
        } else {
            chains.remove(&ctx.hart_id).expect("f32 matmul: continuation without first")
        }
    });
    assert!(chain.rows == rows && chain.bank == c && chain.base == base,
            "f32 matmul: continuation changed destination");
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = chain.values[row * cols + col];
            for inner in 0..k { sum = av[row * k + inner].mul_add(bv[col * k + inner], sum); }
            assert!(sum.is_finite(), "f32 matmul: accumulator overflow");
            chain.values[row * cols + col] = sum;
        }
    }
    if last {
        let pc = pbank(ctx, c);
        for (index, value) in chain.values.iter().enumerate() {
            let offset = base * stride + index * 4;
            ctx.banks[pc][offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    } else {
        CHAINS.with(|chains| { chains.borrow_mut().insert(ctx.hart_id, chain); });
    }
    0
}
