//===- 55_mxfp2int.rs - MXFP2INT instruction (MXFP4 to INT8) ---------------===//
//
// Dequantizes iter MXFP4 blocks of 32 elements to int8.
//
// op1 bank: block b in row b, element e in byte e / 2, low nibble first; the
//           element value is raw - 8.
// op2 bank: E8M0 scale of block b in byte b % 16 of row b / 16.
// wr bank:  elements 0..15 of block b in row 2b, 16..31 in row 2b + 1,
//           saturate(value * 2^(scale - 127)); the right shift rounds toward
//           minus infinity.
// rs2 is reserved and zero.
//
//===----------------------------------------------------------------------===//

use bebop_bemu::thread::banks::bank_size;
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use super::instruction::{BallModel, ExecContext};

pub struct Mxfp2Int;

const ROW_BYTES: usize = 16;
const ELEMENTS: usize = 32;

impl BallModel for Mxfp2Int {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        assert!(
            !rs1_read_old(xs1) && !rs1_update(xs1),
            "mxfp2int: reads no old destination; rs1[29] and rs1[63] must be clear"
        );
        assert!(xs2 == 0, "mxfp2int: rs2 is reserved and must be zero");
        let data = rs1_b0(xs1);
        let scales = rs1_b1(xs1);
        let out = rs1_b2(xs1);
        let blocks = rs1_iter(xs1) as usize;
        for bank in [data, scales, out] {
            let config = ctx.config(bank);
            assert!(config.allocated && config.cols == 1, "mxfp2int: bank {bank} must be one allocated group");
        }
        let lines = bank_size() / ROW_BYTES;
        assert!(blocks > 0 && blocks * 2 <= lines, "mxfp2int: iter must be 1..half the bank depth");

        let data = pbank(ctx, data);
        let scales = pbank(ctx, scales);
        let out = pbank(ctx, out);
        for block in 0..blocks {
            let scale = ctx.banks[scales][(block / 16) * ROW_BYTES + block % 16];
            let shift = scale as i32 - 127;
            let row = &ctx.banks[data][block * ROW_BYTES..(block + 1) * ROW_BYTES];
            let mut result = [0u8; ELEMENTS];
            for (element, byte) in result.iter_mut().enumerate() {
                let value = ((row[element / 2] >> ((element % 2) * 4)) & 0x0f) as i32 - 8;
                let scaled = if shift >= 0 {
                    // |value| <= 8, so shifts past 7 saturate.
                    (value as i64) << shift.min(8)
                } else {
                    (value >> (-shift).min(31)) as i64
                };
                *byte = scaled.clamp(-128, 127) as i8 as u8;
            }
            let start = block * 2 * ROW_BYTES;
            ctx.banks[out][start..start + ELEMENTS].copy_from_slice(&result);
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        rs1_iter(xs1) * 4
    }
}
