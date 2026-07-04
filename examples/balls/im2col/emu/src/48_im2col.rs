use super::super::bank::BANK_NUM;
use super::decode::{pbank, rs1_b0, rs1_b2};
use super::instruction::{ExecContext, Instruction};

pub struct Im2col;

impl Instruction for Im2col {
    const FUNCT: u32 = 48;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let wr = rs1_b2(xs1);

        if op1 >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("im2col: invalid bank_id");
        }
        if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
            panic!("im2col: bank not allocated");
        }
        if op1 == wr {
            panic!("im2col: op1 and wr must differ");
        }

        let kcol = (xs2 & 0xFF) as usize;
        let krow = ((xs2 >> 8) & 0xFF) as usize;
        let incol = ((xs2 >> 16) & 0xFF) as usize;
        let inrow = ((xs2 >> 24) & 0xFF) as usize;
        let startcol = ((xs2 >> 32) & 0xFF) as usize;
        let startrow = ((xs2 >> 40) & 0xFF) as usize;
        let col_step = ((xs2 >> 48) & 0xFF) as usize;

        if kcol == 0 || krow == 0 || incol == 0 || inrow == 0 || col_step == 0 {
            panic!("im2col: invalid shape (zero dim)");
        }
        if incol < kcol || inrow < krow {
            panic!("im2col: kernel larger than input");
        }

        let row_end = inrow - krow;
        let col_end = incol - kcol;
        if startrow > row_end || startcol > col_end {
            panic!("im2col: invalid start window");
        }

        let po = pbank(ctx.bank_map, op1);
        let pw = pbank(ctx.bank_map, wr);
        let (srcb, dstb): (&[u8], &mut [u8]) = if po < pw {
            let (l, r) = ctx.banks.split_at_mut(pw);
            (&l[po], &mut r[0])
        } else {
            let (l, r) = ctx.banks.split_at_mut(po);
            (&r[0], &mut l[pw])
        };

        let mut out = 0usize;
        for r in startrow..=row_end {
            for c in (startcol..=col_end).step_by(col_step) {
                for kr in 0..krow {
                    for kc in 0..kcol {
                        let src = r * incol + c + kr * incol + kc;
                        if src >= srcb.len() || out >= dstb.len() {
                            panic!("im2col: range src={src} out={out}");
                        }
                        dstb[out] = srcb[src];
                        out += 1;
                    }
                }
            }
        }
        0
    }

    fn latency(_xs1: u64, xs2: u64) -> u64 {
        let kcol = xs2 & 0xFF;
        let krow = (xs2 >> 8) & 0xFF;
        let incol = (xs2 >> 16) & 0xFF;
        let inrow = (xs2 >> 24) & 0xFF;
        let startcol = (xs2 >> 32) & 0xFF;
        let startrow = (xs2 >> 40) & 0xFF;
        let col_step = (xs2 >> 48) & 0xFF;

        if kcol == 0 || krow == 0 || incol == 0 || inrow == 0 || col_step == 0 {
            return 16;
        }
        if incol < kcol || inrow < krow {
            return 16;
        }

        let row_end = inrow - krow;
        let col_end = incol - kcol;
        if startrow > row_end || startcol > col_end {
            return 16;
        }

        let col_windows = ((col_end - startcol) / col_step) + 1;
        let nwin = (row_end - startrow + 1).saturating_mul(col_windows);
        nwin.saturating_mul(krow).saturating_mul(kcol).max(16)
    }
}
