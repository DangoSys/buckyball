//===- 55_mxfp2int.rs - MXFP2INT instruction (MXFP to INT8) ----------------===//
//
// Dequantizes MXFP4/6/8 data to INT8 using per-block E8M0 scales from MMIO.
//
// rs1[9:0]:    op1_bank (BANK0, input MXFP data)
// rs1[29:20]:  wr_bank (BANK2, output INT8 data)
// rs1[63:30]:  iter (number of MXFP blocks to process)
//
// For MXFP4: each block = 32 x 4-bit FP = 16 bytes input → 32 bytes INT8 output
// Scale: per-block E8M0 (8-bit) from MMIO, indexed by block_idx
// MMIO region must be bound to wr_bank (BANK2) via bb_mmio_set beforehand.
//
//===-----------------------------------------------------------------===//

use super::super::bank::{mmio_read_byte, BANK_WIDTH};
use super::decode::{rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

pub struct Mxfp2Int;

impl Instruction for Mxfp2Int {
    const FUNCT: u32 = 55;

    fn exec(xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        let in_bank = rs1_b0(xs1) as u32;
        let out_bank = rs1_b2(xs1) as u32;
        let iter = rs1_iter(xs1) as usize; // number of MXFP blocks

        if std::env::var("BEMU_RTRACE").is_ok() {
            eprintln!(
                "[RTRACE] mxfp2int: in_bank={} out_bank={} iter={}",
                in_bank, out_bank, iter
            );
        }

        // Resolve physical banks
        let in_pbank = ctx
            .bank_map
            .resolve(in_bank)
            .unwrap_or_else(|| panic!("mxfp2int: input bank {} not allocated", in_bank));
        let out_pbank = ctx
            .bank_map
            .resolve(out_bank)
            .unwrap_or_else(|| panic!("mxfp2int: output bank {} not allocated", out_bank));

        const FP4_PER_BLOCK: usize = 32;
        const BYTES_PER_OUTPUT_BLOCK: usize = 32; // 32 x INT8

        for block_idx in 0..iter {
            // Read E8M0 scale from MMIO (meta_bank = out_bank, rel_addr = block_idx)
            let scale_e8m0 = mmio_read_byte(ctx.mmio_banks, ctx.mmio_region_table, out_bank as usize, block_idx);

            // Read input MXFP4 block (16 bytes = 1 bank row)
            let in_row_addr = block_idx;
            let in_row_bytes =
                &ctx.banks[in_pbank][in_row_addr * (BANK_WIDTH / 8)..(in_row_addr + 1) * (BANK_WIDTH / 8)];

            // Dequantize 32 FP4 elements to INT8
            let mut out_bytes = [0i8; BYTES_PER_OUTPUT_BLOCK];
            for (elem_idx, out_byte) in out_bytes.iter_mut().enumerate().take(FP4_PER_BLOCK) {
                let byte_idx = elem_idx / 2;
                let nibble_shift = (elem_idx % 2) * 4;
                let fp4_raw = (in_row_bytes[byte_idx] >> nibble_shift) & 0x0F;

                // FP4 decode: signed_val = raw - 8 (range -8..+7)
                let signed_val = (fp4_raw as i32) - 8;

                // Scale: shift = scale_e8m0 - 127
                let shift = (scale_e8m0 as i32) - 127;

                // Apply scale and saturate to INT8
                let scaled = if shift >= 0 {
                    signed_val
                        .checked_shl(shift as u32)
                        .unwrap_or(if signed_val >= 0 { i32::MAX } else { i32::MIN })
                } else {
                    let abs_shift = (-shift) as u32;
                    if abs_shift >= 32 {
                        0 // Shift too large, result is 0
                    } else {
                        signed_val >> abs_shift
                    }
                };
                let saturated = scaled.clamp(-128, 127) as i8;

                *out_byte = saturated;
            }

            // Write output INT8 data (32 bytes = 2 bank rows)
            let out_row_base = block_idx * 2;
            for row_offset in 0..2 {
                let out_row_addr = out_row_base + row_offset;
                let out_row_start = out_row_addr * (BANK_WIDTH / 8);
                let src_start = row_offset * (BANK_WIDTH / 8);
                let src_end = src_start + (BANK_WIDTH / 8);

                // Copy 16 bytes (1 row) from out_bytes to output bank
                for (i, &byte) in out_bytes[src_start..src_end].iter().enumerate() {
                    ctx.banks[out_pbank][out_row_start + i] = byte as u8;
                }
            }
        }

        0
    }

    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        let iter = rs1_iter(_xs1);
        // Approximate: each block needs 1 MMIO read + 1 input read + 2 output writes
        iter * 4
    }
}
