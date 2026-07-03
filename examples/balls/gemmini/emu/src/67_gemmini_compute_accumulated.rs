use super::super::bank::BANK_NUM;
use super::bank_matrix::{read_i32_nn_groups, read_i8_nn, write_i32_nn_groups};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::gemmini_state::gemini;
use super::instruction::{ExecContext, Instruction};

pub struct GemminiComputeAccumulated;

impl Instruction for GemminiComputeAccumulated {
    const FUNCT: u32 = 67;

    fn exec(xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op_a = rs1_b0(xs1);
        let op_b = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let n = rs1_iter(xs1) as usize;

        if op_a >= BANK_NUM as u64 || op_b >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("gemmini_compute_accumulated: invalid bank_id");
        }
        if !ctx.cfgs[op_a as usize].allocated || !ctx.cfgs[op_b as usize].allocated || !ctx.cfgs[wr as usize].allocated
        {
            panic!("gemmini_compute_accumulated: bank not allocated");
        }
        if n == 0 || n > 64 {
            panic!("gemmini_compute_accumulated: bad iter");
        }

        let pa = pbank(ctx.bank_map, op_a);
        let pb = pbank(ctx.bank_map, op_b);
        let pw: Vec<_> = (0..ctx.cfgs[wr as usize].cols)
            .map(|group| pbank_group(ctx.bank_map, wr, group))
            .collect();

        let gm = gemini().lock().unwrap();
        let a_transpose = gm.cfg.a_transpose;
        let b_transpose = gm.cfg.b_transpose;
        let in_shift = gm.cfg.in_shift;
        drop(gm);

        let a = read_i8_nn(ctx.banks, pa, n);
        let b = read_i8_nn(ctx.banks, pb, n);
        let mut c = read_i32_nn_groups(ctx.banks, &pw, n);

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let av = if a_transpose { a[k][i] } else { a[i][k] };
                    let bv = if b_transpose { b[j][k] } else { b[k][j] };
                    c[i][j] += av as i32 * bv as i32;
                }
                if in_shift > 0 {
                    c[i][j] >>= in_shift;
                }
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
