//===- 64_mul_warp16.rs - MUL_WARP16 instruction ---------------------------===//

use super::super::bank::BANK_NUM;
use super::bank_matrix::{read_i32_nn_groups, read_i8_k_rows, write_i32_nn_groups};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

const WARP_M: usize = 16;
const WARP_N: usize = 16;

pub struct MulWarp16;

impl Instruction for MulWarp16 {
    const FUNCT: u32 = 64;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let op2 = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let iter = rs1_iter(xs1);
        let _ = xs2;

        if std::env::var("BEMU_RTRACE").is_ok() {
            eprintln!("[RTRACE] mul_warp16: banks[{},{},{}] iter={}", op1, op2, wr, iter);
        }

        if op1 >= BANK_NUM as u64 || op2 >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("mul_warp16: invalid bank_id");
        }

        let c1 = ctx.cfgs[op1 as usize].cols;
        let c2 = ctx.cfgs[op2 as usize].cols;
        let cw = ctx.cfgs[wr as usize].cols;
        if c1 != 1 || c2 != 1 || cw != 4 {
            panic!("mul_warp16: unsupported bank layout op1_cols={c1} op2_cols={c2} wr_cols={cw}");
        }

        let p1 = pbank(ctx.bank_map, op1);
        let p2 = pbank(ctx.bank_map, op2);
        let pw: Vec<_> = (0..cw).map(|group| pbank_group(ctx.bank_map, wr, group)).collect();
        let kin = iter as usize;

        if kin == 0 {
            panic!("mul_warp16: iter must be > 0");
        }

        let need = kin * WARP_N;
        if need > ctx.banks[p1].len() || need > ctx.banks[p2].len() {
            panic!("mul_warp16: iter too large for bank");
        }

        let a_t = read_i8_k_rows(ctx.banks, p1, kin, WARP_N);
        let b = read_i8_k_rows(ctx.banks, p2, kin, WARP_N);
        let mut c = read_i32_nn_groups(ctx.banks, &pw, 16);

        for i in 0..WARP_M {
            for j in 0..WARP_N {
                let mut acc = c[i][j];
                for t in 0..kin {
                    acc = acc.wrapping_add((a_t[t][i] as i32).wrapping_mul(b[t][j] as i32));
                }
                c[i][j] = acc;
            }
        }

        write_i32_nn_groups(ctx.banks, &pw, &c, 16);
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let kin = rs1_iter(xs1).max(1);
        kin.saturating_mul(16)
    }
}
