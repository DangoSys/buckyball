//===- 64_vecmat16.rs - VECMAT16 instruction ------------------------------===//
//
// C += A^T * B: op1 holds A^T and op2 holds B, iter rows of sixteen int8 each;
// the four-group destination holds C, sixteen rows of sixteen int32, and is
// read as the old value (rs1[29]). rs2 is reserved and zero.
//
//===----------------------------------------------------------------------===//

use super::bank_matrix::{read_i32_nn_groups, read_i8_k_rows, write_i32_nn_groups};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use super::instruction::{BallModel, ExecContext};

const WARP_M: usize = 16;
const WARP_N: usize = 16;

pub struct VecMat16;

impl BallModel for VecMat16 {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        assert!(rs1_read_old(xs1), "vecmat16: accumulates into the old destination; rs1[29] must be set");
        assert!(xs2 == 0, "vecmat16: rs2 is reserved and must be zero");
        let op1 = rs1_b0(xs1);
        let op2 = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let iter = rs1_iter(xs1);

        let c1 = ctx.config(op1).cols;
        let c2 = ctx.config(op2).cols;
        let cw = ctx.config(wr).cols;
        if c1 != 1 || c2 != 1 || cw != 4 {
            panic!("vecmat16: unsupported bank layout op1_cols={c1} op2_cols={c2} wr_cols={cw}");
        }

        let p1 = pbank(ctx, op1);
        let p2 = pbank(ctx, op2);
        let pw: Vec<_> = (0..cw).map(|group| pbank_group(ctx, wr, group)).collect();
        let kin = iter as usize;

        if kin == 0 {
            panic!("vecmat16: iter must be > 0");
        }

        let need = kin * WARP_N;
        if need > ctx.banks[p1].len() || need > ctx.banks[p2].len() {
            panic!("vecmat16: iter too large for bank");
        }

        let a_t = read_i8_k_rows(&ctx.banks, p1, kin, WARP_N);
        let b = read_i8_k_rows(&ctx.banks, p2, kin, WARP_N);
        let mut c = read_i32_nn_groups(&ctx.banks, &pw, 16);

        for i in 0..WARP_M {
            for j in 0..WARP_N {
                let mut acc = c[i][j];
                for t in 0..kin {
                    acc = acc.wrapping_add((a_t[t][i] as i32).wrapping_mul(b[t][j] as i32));
                }
                c[i][j] = acc;
            }
        }

        write_i32_nn_groups(&mut ctx.banks, &pw, &c, 16);
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let kin = rs1_iter(xs1).max(1);
        kin.saturating_mul(16)
    }
}
