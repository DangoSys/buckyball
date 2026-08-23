//===- fp2int.rs - FP32 to INT8 activation quantization -------------------===//

use super::super::bank::{bank_num, bank_size, mmio_total_size, mmio_write_byte};
use super::decode::{pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub(crate) fn exec_fp2int(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let src = rs1_b0(xs1);
    let dst = rs1_b2(xs1);
    let depth = rs1_iter(xs1) as usize;

    if src >= bank_num() as u64 || dst >= bank_num() as u64 {
        panic!("fp2int: invalid bank_id");
    }
    if src == dst {
        panic!("fp2int: in-place quantization is forbidden");
    }

    if depth == 0 {
        panic!("fp2int: iter must be > 0");
    }

    let sc = ctx.cfgs[src as usize];
    let dc = ctx.cfgs[dst as usize];
    if !sc.allocated || !dc.allocated {
        panic!("fp2int: bank not allocated");
    }

    if xs2 >> 13 != 0 {
        panic!("fp2int: reserved rs2 bits are nonzero");
    }
    let scale_addr = (xs2 & 0x1fff) as usize;
    if scale_addr != 0 {
        panic!("fp2int: activation scale address must be 0");
    }
    if scale_addr + 4 > mmio_total_size() {
        panic!("fp2int: activation scale address out of range");
    }

    if sc.cols == 0 || dc.cols == 0 {
        panic!("fp2int: source and destination groups must be nonzero");
    }
    let source_words = depth * sc.cols as usize;
    let destination_row_words = 4 * dc.cols as usize;
    if source_words % destination_row_words != 0 {
        panic!("fp2int: source stream does not fill destination rows");
    }

    let mut max_abs = 0u32;
    for row in 0..depth {
        let base = row * 16;
        if base + 16 > bank_size() {
            panic!("fp2int: source out of range");
        }
        for group in 0..sc.cols {
            let ps = pbank_group(ctx.bank_map, src, group);
            for lane in 0..4 {
                let off = base + lane * 4;
                let bits = u32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                if (bits >> 23) & 0xff == 0xff {
                    panic!("fp2int: activation contains NaN or infinity");
                }
                max_abs = max_abs.max(bits & 0x7fff_ffff);
            }
        }
    }
    let da_bits = super::model::fp2int_da_from_max_abs_bits(max_abs);
    for (i, byte) in da_bits.to_le_bytes().iter().enumerate() {
        mmio_write_byte(ctx.mmio_banks, scale_addr + i, *byte);
    }
    let quant_scale_bits = super::model::fp32_divide(1.0f32.to_bits(), da_bits);
    let mut packed = [0u8; 16];
    let mut source_word = 0usize;
    for row in 0..depth {
        let base = row * 16;
        for group in 0..sc.cols {
            let ps = pbank_group(ctx.bank_map, src, group);
            let byte_base = (source_word % 4) * 4;
            for lane in 0..4 {
                let off = base + lane * 4;
                let bits = u32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                packed[byte_base + lane] = super::model::fp2int_i8_bits(bits, quant_scale_bits) as u8;
            }
            source_word += 1;
            if source_word % 4 == 0 {
                let output_word = source_word / 4 - 1;
                let output_group = output_word % dc.cols as usize;
                let output_row = output_word / dc.cols as usize;
                let pd = pbank_group(ctx.bank_map, dst, output_group as u64);
                let output_base = output_row * 16;
                if output_base + 16 > bank_size() {
                    panic!("fp2int: destination out of range");
                }
                ctx.banks[pd][output_base..output_base + 16].copy_from_slice(&packed);
            }
        }
    }
    0
}

pub(crate) fn latency(xs1: u64) -> u64 {
    rs1_iter(xs1).max(1)
}
