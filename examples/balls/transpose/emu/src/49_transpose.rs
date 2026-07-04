use super::super::bank::BANK_NUM;
use super::decode::{pbank, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

const TRANSPOSE_M: usize = 16;

pub struct Transpose;

impl Instruction for Transpose {
    const FUNCT: u32 = 49;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let wr = rs1_b2(xs1);
        let iter = rs1_iter(xs1);
        let _ = xs2;

        if std::env::var("BEMU_RTRACE").is_ok() {
            eprintln!("[RTRACE] transpose: bank{} -> bank{} iter={}", op1, wr, iter);
        }

        if op1 >= BANK_NUM as u64 || wr >= BANK_NUM as u64 {
            panic!("transpose: invalid bank_id");
        }

        if iter == 0 {
            panic!("transpose: iter must be > 0");
        }

        let c1 = ctx.cfgs[op1 as usize].cols;
        let cw = ctx.cfgs[wr as usize].cols;
        if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
            panic!("transpose: bank not allocated");
        }

        let k = iter as usize;
        let po = pbank(ctx.bank_map, op1);
        let pw = pbank(ctx.bank_map, wr);

        if c1 == 1 && cw == 1 {
            if po == pw {
                panic!("transpose: op1 and wr must differ");
            }

            let (srcb, dstb): (&[u8], &mut [u8]) = if po < pw {
                let (l, r) = ctx.banks.split_at_mut(pw);
                (&l[po], &mut r[0])
            } else {
                let (l, r) = ctx.banks.split_at_mut(po);
                (&r[0], &mut l[pw])
            };

            for r in 0..TRANSPOSE_M {
                for c in 0..k {
                    let src = r * k + c;
                    let dst = c * TRANSPOSE_M + r;
                    if src >= srcb.len() || dst >= dstb.len() {
                        panic!("transpose: bank range src={src} dst={dst}");
                    }
                    dstb[dst] = srcb[src];
                }
            }
            return 0;
        }

        if c1 == 4 && cw == 4 {
            let n = k;
            for i in 0..n {
                for j in 0..n {
                    let src_off = i * 64 + j * 4;
                    let dst_off = j * 64 + i * 4;
                    let v = i32::from_le_bytes(ctx.banks[po][src_off..src_off + 4].try_into().unwrap());
                    ctx.banks[pw][dst_off..dst_off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            return 0;
        }

        panic!("transpose: unsupported bank layout op1_cols={c1} wr_cols={cw}");
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        let k = rs1_iter(xs1).clamp(1, 64);
        k.saturating_mul(k)
    }
}
