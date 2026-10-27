use bebop_bemu::thread::banks::bank_size;
use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use super::instruction::{BallModel, ExecContext};

pub struct Relu;

impl BallModel for Relu {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let bank = rs1_b0(xs1);
        let group = rs1_b1(xs1);
        let iter = rs1_iter(xs1) as usize;
        let stride = xs2 as usize;
        // The RTL requires op1 and wr to name one renamed bank: only an in-place
        // update (rs1[63]) keeps the destination alias.
        assert!(
            !rs1_read_old(xs1) && rs1_update(xs1),
            "relu: updates its bank in place; rs1[63] must be set and rs1[29] clear"
        );

        if rs1_b2(xs1) != bank {
            panic!("relu: source and destination banks must match");
        }
        let config = *ctx.config(bank);
        if !config.allocated {
            panic!("relu: bank not allocated");
        }
        if group >= config.cols as u64 {
            panic!("relu: group is not allocated in bank");
        }
        let lines = bank_size() / 16;
        if iter == 0 || iter % 16 != 0 || iter > stride {
            panic!("relu: iter must be a 16-aligned positive value no larger than stride");
        }
        if stride == 0 || lines % stride != 0 {
            panic!("relu: stride must divide the physical bank depth");
        }

        let physical = pbank_group(ctx, bank, group);
        for segment in (0..lines).step_by(stride) {
            for line in 0..iter {
                let offset = (segment + line) * 16;
                for lane in 0..4 {
                    let byte = offset + lane * 4;
                    let value =
                        i32::from_le_bytes(ctx.banks[physical][byte..byte + 4].try_into().unwrap());
                    ctx.banks[physical][byte..byte + 4]
                        .copy_from_slice(&value.max(0).to_le_bytes());
                }
            }
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        let stride = xs2;
        let lines = bank_size() as u64 / 16;
        if iter == 0 || iter % 16 != 0 || iter > stride || stride == 0 || lines % stride != 0 {
            panic!("relu: invalid iter or stride");
        }
        lines / stride * iter * 4
    }
}
