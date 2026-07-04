use super::super::bank::BANK_NUM;
use super::bank_matrix::{read_i32_nn, read_i8_nn, write_i32_nn_groups};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::gemmini_state::gemini;
use super::instruction::{ExecContext, Instruction};

pub struct GemminiComputePreloaded;

impl Instruction for GemminiComputePreloaded {
    const FUNCT: u32 = 66;

    fn exec(xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op_a = rs1_b0(xs1);
        let op_b = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let n = rs1_iter(xs1) as usize;

        if op_a >= BANK_NUM as u64 || op_b >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("gemmini_compute_preloaded: invalid bank_id");
        }
        if !ctx.cfgs[op_a as usize].allocated || !ctx.cfgs[op_b as usize].allocated || !ctx.cfgs[wr as usize].allocated
        {
            panic!("gemmini_compute_preloaded: bank not allocated");
        }
        if n == 0 || n > 64 {
            panic!("gemmini_compute_preloaded: bad iter");
        }

        let pa = pbank(ctx.bank_map, op_a);
        let pb = pbank(ctx.bank_map, op_b);
        let pw: Vec<_> = (0..ctx.cfgs[wr as usize].cols)
            .map(|group| pbank_group(ctx.bank_map, wr, group))
            .collect();
        let gm = gemini().lock().unwrap();
        let df = gm.cfg.dataflow;
        let a_transpose = gm.cfg.a_transpose;
        let b_transpose = gm.cfg.b_transpose;
        let in_shift = gm.cfg.in_shift;
        let ws_b = gm.ws_b.clone();
        drop(gm);

        if df == 1 {
            let b = ws_b.expect("gemmini_compute_preloaded: WS missing preload");
            let a = read_i8_nn(ctx.banks, pa, n);
            let d = read_i32_nn(ctx.banks, pb, n);
            let mut c = vec![vec![0i32; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = d[i][j];
                    for k in 0..n {
                        acc += a[i][k] as i32 * b[k][j] as i32;
                    }
                    c[i][j] = acc;
                }
            }
            write_i32_nn_groups(ctx.banks, &pw, &c, n);
        } else {
            // OS mode: per RTL GemminiExCtrlPreloadStates, preload feeds D=0 to
            // mesh in OS mode, so the accumulator starts at zero.
            let a = read_i8_nn(ctx.banks, pa, n);
            let b = read_i8_nn(ctx.banks, pb, n);
            let mut c = vec![vec![0i32; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0i32;
                    for k in 0..n {
                        let av = if a_transpose { a[k][i] } else { a[i][k] };
                        let bv = if b_transpose { b[j][k] } else { b[k][j] };
                        acc += av as i32 * bv as i32;
                    }
                    if in_shift > 0 {
                        acc >>= in_shift;
                    }
                    c[i][j] = acc;
                }
            }
            write_i32_nn_groups(ctx.banks, &pw, &c, n);
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let n = rs1_iter(xs1).clamp(1, 64);
        n.saturating_mul(n).saturating_mul(n) / 4 + n.saturating_mul(n)
    }
}
