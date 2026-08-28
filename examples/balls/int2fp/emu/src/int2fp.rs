//===- int2fp.rs - INT32 accumulator dequantization -----------------------===//
//
// FP32 output = INT32 accumulator * Da * Dw.
// Da and Dw are FP32 values stored in the unified MMIO byte address space.

use super::super::bank::{bank_num, bank_size, mmio_read_byte, mmio_total_size};
use super::decode::{pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

fn read_f32(mmio_banks: &[Vec<u8>], addr: usize) -> f32 {
    if addr % 4 != 0 {
        panic!("int2fp: scale address must be 4-byte aligned");
    }
    if addr + 4 > mmio_total_size() {
        panic!("int2fp: scale address out of range");
    }
    let bytes = [
        mmio_read_byte(mmio_banks, addr),
        mmio_read_byte(mmio_banks, addr + 1),
        mmio_read_byte(mmio_banks, addr + 2),
        mmio_read_byte(mmio_banks, addr + 3),
    ];
    let scale = f32::from_le_bytes(bytes);
    if !scale.is_finite() || scale <= 0.0 {
        panic!("int2fp: scale must be finite and positive");
    }
    scale
}

pub(crate) fn exec_int2fp(xs1: u64, xs2: u64, ctx: &mut ExecContext, per_channel: bool) -> u64 {
    let src = rs1_b0(xs1);
    let dst = rs1_b2(xs1);
    let depth = rs1_iter(xs1) as usize;
    let act_scale_addr = (xs2 & 0x1fff) as usize;
    let weight_scale_addr = ((xs2 >> 13) & 0x1fff) as usize;
    if act_scale_addr != 0 {
        panic!("int2fp: activation scale address must be 0");
    }
    if weight_scale_addr < 16 {
        panic!("int2fp: weight scale address must be >= 16");
    }
    if xs2 >> 26 != 0 {
        panic!("int2fp: reserved rs2 bits are nonzero");
    }
    if src >= bank_num() as u64 || dst >= bank_num() as u64 {
        panic!("int2fp: invalid bank_id");
    }
    if src == dst {
        panic!("int2fp: in-place dequantization is forbidden");
    }
    if depth == 0 {
        panic!("int2fp: iter must be > 0");
    }
    let sc = ctx.cfgs[src as usize];
    let dc = ctx.cfgs[dst as usize];
    if !sc.allocated || !dc.allocated {
        panic!("int2fp: bank not allocated");
    }
    if sc.cols != dc.cols || sc.cols == 0 || sc.cols > 4 {
        panic!("int2fp: requires matching 1..4 INT32/FP32 bank groups");
    }
    if per_channel && weight_scale_addr + sc.cols as usize * 16 > mmio_total_size() {
        panic!("int2fp: channel scale range out of range");
    }

    let da = read_f32(ctx.mmio_banks, act_scale_addr);
    let tensor_dw = (!per_channel).then(|| read_f32(ctx.mmio_banks, weight_scale_addr));
    for group in 0..sc.cols as u64 {
        let ps = pbank_group(ctx.bank_map, src, group);
        let pd = pbank_group(ctx.bank_map, dst, group);
        for row in 0..depth {
            let base = row * 16;
            if base + 16 > bank_size() {
                panic!("int2fp: bank row out of range");
            }
            for lane in 0..4usize {
                let off = base + lane * 4;
                let acc = i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                let dw = tensor_dw.unwrap_or_else(|| {
                    read_f32(
                        ctx.mmio_banks,
                        weight_scale_addr + (group as usize * 4 + lane) * 4,
                    )
                });
                let out = (acc as f32) * da * dw;
                ctx.banks[pd][off..off + 4].copy_from_slice(&out.to_le_bytes());
            }
        }
    }
    0
}
pub(crate) fn latency(xs1: u64) -> u64 {
    rs1_iter(xs1).max(1)
}
