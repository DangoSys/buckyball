use super::bank_matrix::read_i8_nn_at;
use super::decode::{pbank, rs1_b0, rs1_b2, rs1_iter};
use super::gemmini_state::gemini;
use super::instruction::{BallInstruction, ExecContext};

pub struct GemminiPreload;

impl BallInstruction for GemminiPreload {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let wr = rs1_b2(xs1);
        let n = rs1_iter(xs1) as usize;

        if !ctx.config(op1).allocated || !ctx.config(wr).allocated {
            panic!("gemmini_preload: bank not allocated");
        }
        if n == 0 || n > 64 {
            panic!("gemmini_preload: bad iter");
        }

        let p1 = pbank(ctx, op1);
        let mut gm = gemini().lock().unwrap();

        if gm.cfg.dataflow == 1 {
            let b = read_i8_nn_at(&ctx.banks, p1, ((xs2 >> 6) & 0x3ff) as usize, n);
            gm.ws_b = Some(if gm.cfg.b_transpose {
                (0..n).map(|i| (0..n).map(|j| b[j][i]).collect()).collect()
            } else {
                b
            });
        } else {
            // RTL OS preload only primes the mesh and injects D=0. It does
            // not write the destination SRAM bank; compute produces C.
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let n = rs1_iter(xs1).clamp(1, 64);
        n.saturating_mul(n)
    }
}
