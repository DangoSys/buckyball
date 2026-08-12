use super::super::bank::bank_num;
use super::bank_matrix::read_i8_nn;
use super::decode::{pbank, rs1_b0, rs1_b2, rs1_iter};
use super::gemmini_state::gemini;
use super::instruction::{ExecContext, Instruction};

pub struct GemminiPreload;

impl Instruction for GemminiPreload {
    const FUNCT: u32 = 53;

    fn exec(xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let wr = rs1_b2(xs1);
        let n = rs1_iter(xs1) as usize;

        if op1 >= bank_num() as u64 || wr >= bank_num() as u64 {
            panic!("gemmini_preload: invalid bank_id");
        }
        if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
            panic!("gemmini_preload: bank not allocated");
        }
        if n == 0 || n > 64 {
            panic!("gemmini_preload: bad iter");
        }

        let p1 = pbank(ctx.bank_map, op1);
        let mut gm = gemini().lock().unwrap();

        if gm.cfg.dataflow == 1 {
            let b = read_i8_nn(&ctx.banks, p1, n);
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
