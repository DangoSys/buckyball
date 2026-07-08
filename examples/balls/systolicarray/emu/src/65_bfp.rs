use super::super::bank::BANK_NUM;
use super::bank_matrix::{read_i8_nn, write_i32_nn_groups};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

pub struct Bfp;

impl Instruction for Bfp {
    const FUNCT: u32 = 65;

    fn exec(xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let op2 = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let n = rs1_iter(xs1) as usize;

        if op1 >= BANK_NUM as u64 || op2 >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("bfp: invalid bank_id");
        }
        if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[op2 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
            panic!("bfp: bank not allocated");
        }
        if ctx.cfgs[wr as usize].cols != 4 {
            panic!("bfp: wr bank must be acc (cols=4)");
        }
        if n == 0 || n > 64 {
            panic!("bfp: bad iter");
        }

        let p1 = pbank(ctx.bank_map, op1);
        let p2 = pbank(ctx.bank_map, op2);
        let pw: Vec<_> = (0..ctx.cfgs[wr as usize].cols)
            .map(|group| pbank_group(ctx.bank_map, wr, group))
            .collect();

        let a = read_i8_nn(ctx.banks, p1, n);
        let b = read_i8_nn(ctx.banks, p2, n);
        let mut c = vec![vec![0i32; n]; n];

        for i in 0..n {
            for j in 0..n {
                let mut acc = 0i32;
                for k in 0..n {
                    acc += a[i][k] as i32 * b[k][j] as i32;
                }
                c[i][j] = acc;
            }
        }

        write_i32_nn_groups(ctx.banks, &pw, &c, n);
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let n = rs1_iter(xs1).clamp(1, 64);
        n.saturating_mul(n).saturating_mul(n) / 4 + n.saturating_mul(n)
    }
}
