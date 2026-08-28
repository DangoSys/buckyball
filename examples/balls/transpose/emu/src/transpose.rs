use super::super::bank::{bank_num, bank_row_bytes, bank_size, bank_width};
use super::decode::{pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub(crate) fn exec_transpose(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let op1 = rs1_b0(xs1);
    let wr = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    let elem_bits = (xs2 & 0xff) as usize;

    if xs2 >> 8 != 0 {
        panic!("transpose: rs2[63:8] must be 0");
    }
    if op1 >= bank_num() as u64 || wr >= bank_num() as u64 {
        panic!("transpose: invalid bank_id");
    }
    if op1 == wr {
        panic!("transpose: op1 and wr must differ");
    }
    if iter == 0 {
        panic!("transpose: iter must be > 0");
    }
    if elem_bits != 8 && elem_bits != 32 {
        panic!("transpose: unsupported elem_bits={elem_bits}");
    }
    if bank_width() % elem_bits != 0 {
        panic!(
            "transpose: bank_width={} not divisible by elem_bits={}",
            bank_width(),
            elem_bits
        );
    }

    let c1 = ctx.cfgs[op1 as usize].cols as usize;
    let cw = ctx.cfgs[wr as usize].cols as usize;
    if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
        panic!("transpose: bank not allocated");
    }
    if c1 == 0 || c1 != cw {
        panic!("transpose: cols mismatch op1_cols={c1} wr_cols={cw}");
    }

    let n = c1;
    let row_bytes = bank_row_bytes();
    let elem_bytes = elem_bits / 8;
    let epg = row_bytes / elem_bytes;
    let w = n * epg;
    let total = iter * w;

    if iter * row_bytes > bank_size() {
        panic!("transpose: src depth out of range iter={iter}");
    }
    // Dest packs W×iter densely into virt rows of width W → depth = iter.
    if iter * row_bytes > bank_size() {
        panic!("transpose: dst depth out of range iter={iter}");
    }

    let mut src = vec![0u8; total * elem_bytes];
    for r in 0..iter {
        for g in 0..n {
            let p = pbank_group(ctx.bank_map, op1, g as u64);
            let bank_off = r * row_bytes;
            let flat_off = (r * w + g * epg) * elem_bytes;
            src[flat_off..flat_off + row_bytes]
                .copy_from_slice(&ctx.banks[p][bank_off..bank_off + row_bytes]);
        }
    }

    let mut dst = vec![0u8; total * elem_bytes];
    for r in 0..iter {
        for c in 0..w {
            let s = (r * w + c) * elem_bytes;
            let d = (c * iter + r) * elem_bytes;
            dst[d..d + elem_bytes].copy_from_slice(&src[s..s + elem_bytes]);
        }
    }

    for idx in 0..total {
        let byte_off = idx * elem_bytes;
        let virt_row = idx / w;
        let virt_col = idx % w;
        let g = virt_col / epg;
        let lane = virt_col % epg;
        let p = pbank_group(ctx.bank_map, wr, g as u64);
        let bank_off = virt_row * row_bytes + lane * elem_bytes;
        if bank_off + elem_bytes > bank_size() {
            panic!("transpose: dst bank range idx={idx}");
        }
        ctx.banks[p][bank_off..bank_off + elem_bytes]
            .copy_from_slice(&dst[byte_off..byte_off + elem_bytes]);
    }

    0
}

pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    let iter = rs1_iter(xs1).max(1);
    let elem_bits = (xs2 & 0xff).max(1);
    let epg = ((bank_width() as u64) / elem_bits).max(1);
    iter.saturating_mul(epg).saturating_mul(2)
}
