use super::bank_matrix::{read_i32_nn_groups_at, read_i8_nn_at, write_i32_nn_groups_at};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::gemmini_state::{gemini, in_shift as apply_in_shift};
use super::instruction::{BallInstruction, ExecContext};

pub struct GemminiComputeAccumulated;

impl BallInstruction for GemminiComputeAccumulated {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op_a = rs1_b0(xs1);
        let op_b = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let n = rs1_iter(xs1) as usize;

        if !ctx.config(op_a).allocated || !ctx.config(op_b).allocated || !ctx.config(wr).allocated {
            panic!("gemmini_compute_accumulated: bank not allocated");
        }
        if n == 0 || n > 64 {
            panic!("gemmini_compute_accumulated: bad iter");
        }

        let pa = pbank(ctx, op_a);
        let pb = pbank(ctx, op_b);
        let pw: Vec<_> = (0..ctx.config(wr).cols)
            .map(|group| pbank_group(ctx, wr, group))
            .collect();

        let gm = gemini().lock().unwrap();
        let a_transpose = gm.cfg.a_transpose;
        let b_transpose = gm.cfg.b_transpose;
        let shift = gm.cfg.in_shift;
        drop(gm);

        let zero_op1_tail = ((xs2 >> 5) & 1) != 0;
        let a_base = ((xs2 >> 6) & 0x3ff) as usize;
        let b_base = ((xs2 >> 16) & 0x3ff) as usize;
        let out_base = ((xs2 >> 26) & 0x3ff) as usize;
        let op1_valid_rows = ctx.config(op_a).valid_rows.min(n as u64) as usize;
        let mut a = read_i8_nn_at(&ctx.banks, pa, a_base, n);
        if zero_op1_tail {
            for row in &mut a[op1_valid_rows..] {
                row.fill(0);
            }
        }
        let b = read_i8_nn_at(&ctx.banks, pb, b_base, n);
        let mut c = read_i32_nn_groups_at(&ctx.banks, &pw, out_base, n);

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let av = if a_transpose { a[k][i] } else { a[i][k] };
                    let bv = if b_transpose { b[j][k] } else { b[k][j] };
                    c[i][j] += av as i32 * bv as i32;
                }
                c[i][j] = apply_in_shift(c[i][j], shift);
            }
        }

        write_i32_nn_groups_at(&mut ctx.banks, &pw, out_base, &c, n);
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let n = rs1_iter(xs1).clamp(1, 64);
        n.saturating_mul(n).saturating_mul(n) / 4 + n.saturating_mul(n)
    }
}
