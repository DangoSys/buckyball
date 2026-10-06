use super::super::bank::{bank_lines, bank_row_bytes};
use super::accumulator::{Chain, Format, CHAINS};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub(crate) fn execute(xs1: u64, xs2: u64, fused: bool, ctx: &mut ExecContext) -> u64 {
    let format = if fused { Format::Fma32 } else { Format::F32 };
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
        xs2 >> 32 == 0 && (rows == 1 || (rows > 0 && rows % 16 == 0)) && cols > 0 && cols % 16 == 0 && k > 0 && k % 4 == 0,
        "f32 matmul: requires M=1/multiple of 16, N multiple of 16, positive K multiple of four and zero reserved bits"
    );
    assert!(a != b && b != c && a != c, "f32 matmul: banks must differ");
    for bank in [a, b, c] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "f32 matmul: banks must be allocated with one column"
        );
    }
    let stride = bank_row_bytes();
    assert!(
        stride == 16
            && rows * k * 4 <= bank_lines() * stride
            && cols * k * 4 <= bank_lines() * stride
            && base * stride + rows * cols * 4 <= bank_lines() * stride,
        "f32 matmul: bank footprint exceeds depth"
    );
    let pa = pbank(ctx, a);
    let pb = pbank(ctx, b);
    let av = &ctx.banks[pa][..rows * k * 4];
    let bv = &ctx.banks[pb][..cols * k * 4];
    assert!(
        av.chunks_exact(4)
            .chain(bv.chunks_exact(4))
            .all(|x| f32::from_le_bytes(x.try_into().unwrap()).is_finite()),
        "f32 matmul: non-finite operand"
    );
    let mut chain = CHAINS.with(|chains| {
        let mut chains = chains.borrow_mut();
        if first {
            assert!(
                !chains.contains_key(&ctx.hart_id),
                "f32 matmul: first while chain is live"
            );
            Chain {
                format,
                rows,
                cols,
                bank: c,
                base,
                values: vec![0f32; rows * cols],
            }
        } else {
            chains
                .remove(&ctx.hart_id)
                .expect("f32 matmul: continuation without first")
        }
    });
    assert!(
        chain.format == format
            && chain.rows == rows
            && chain.cols == cols
            && chain.bank == c
            && chain.base == base,
        "f32 matmul: continuation changed destination"
    );
    let weights: Vec<&[u8]> = (0..cols)
        .map(|col| &bv[col * k * 4..(col + 1) * k * 4])
        .collect();
    for row in 0..rows {
        if av[row * k * 4..(row + 1) * k * 4]
            .chunks_exact(4)
            .all(|x| u32::from_le_bytes(x.try_into().unwrap()) & 0x7fffffff == 0)
            && chain.values[row * cols..(row + 1) * cols]
                .iter()
                .all(|x| x.to_bits() == 0)
        {
            continue;
        }
        let av = &av[row * k * 4..(row + 1) * k * 4];
        let dst = &mut chain.values[row * cols..(row + 1) * cols];
        let mut sums = dst.to_vec();
        for inner in 0..k {
            let a = f32::from_le_bytes(av[inner * 4..inner * 4 + 4].try_into().unwrap());
            for col in 0..cols {
                let b =
                    f32::from_le_bytes(weights[col][inner * 4..inner * 4 + 4].try_into().unwrap());
                sums[col] = if fused {
                    a.mul_add(b, sums[col])
                } else {
                    (a * b) + sums[col]
                };
            }
        }
        assert!(
            sums.iter().all(|x| x.is_finite()),
            "f32 matmul: accumulator overflow"
        );
        dst.copy_from_slice(&sums);
    }
    if last {
        let pc = pbank(ctx, c);
        for (index, value) in chain.values.iter().enumerate() {
            let offset = base * stride + index * 4;
            ctx.banks[pc][offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    } else {
        CHAINS.with(|chains| {
            chains.borrow_mut().insert(ctx.hart_id, chain);
        });
    }
    0
}
