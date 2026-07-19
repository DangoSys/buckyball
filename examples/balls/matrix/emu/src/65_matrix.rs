use super::super::bank::BANK_NUM;
use super::bank_matrix::read_i8_k_rows;
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2};
use super::instruction::{ExecContext, Instruction};

pub struct Matrix;

impl Instruction for Matrix {
    const FUNCT: u32 = 65;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let op2 = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let m = (xs2 & 0xfff) as usize;
        let n = ((xs2 >> 12) & 0xfff) as usize;
        let k = ((xs2 >> 24) & 0xfff) as usize;

        if op1 >= BANK_NUM as u64 || op2 >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("matrix: invalid bank_id");
        }
        if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[op2 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
            panic!("matrix: bank not allocated");
        }
        if ctx.cfgs[wr as usize].cols != 4 {
            panic!("matrix: wr bank must be acc (cols=4)");
        }
        if m == 0 || n == 0 || k == 0 {
            panic!("matrix: M/N/K must be non-zero");
        }
        if m > 64 || n > 16 || k > 64 {
            panic!("matrix: dimensions exceed bank layout");
        }

        let p1 = pbank(ctx.bank_map, op1);
        let p2 = pbank(ctx.bank_map, op2);
        let pw: Vec<_> = (0..ctx.cfgs[wr as usize].cols)
            .map(|group| pbank_group(ctx.bank_map, wr, group))
            .collect();

        let a = read_i8_k_rows(ctx.banks, p1, m, k);
        let b = read_i8_k_rows(ctx.banks, p2, k, n);
        let mut c = vec![vec![0i32; n]; m];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += a[i][kk] as i32 * b[kk][j] as i32;
                }
                c[i][j] = acc;
            }
        }

        for (i, row) in c.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let group = j / 4;
                let lane = j % 4;
                let off = i * 16 + lane * 4;
                ctx.banks[pw[group]][off..off + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        0
    }

    fn latency(_xs1: u64, xs2: u64) -> u64 {
        let m = (xs2 & 0xfff).clamp(1, 64);
        let n = ((xs2 >> 12) & 0xfff).clamp(1, 16);
        let k = ((xs2 >> 24) & 0xfff).clamp(1, 64);
        m.saturating_mul(n).saturating_mul(k) / 4 + m.saturating_mul(n)
    }
}
