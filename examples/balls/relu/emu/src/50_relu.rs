use super::super::bank::{bank_num, bank_size};
use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{BallInstruction, ExecContext};

pub struct Relu;

impl BallInstruction for Relu {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let bank = rs1_b0(xs1);
        let group = rs1_b1(xs1);
        let iter = rs1_iter(xs1) as usize;
        let stride = xs2 as usize;

        if rs1_b2(xs1) != 0 {
            panic!("relu: bank2 must be zero");
        }
        if bank >= bank_num() as u64 {
            panic!("relu: invalid bank id");
        }
        let config = ctx.cfgs[bank as usize];
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

        let physical = pbank_group(ctx.bank_map, bank, group);
        for segment in (0..lines).step_by(stride) {
            for line in 0..iter {
                let offset = (segment + line) * 16;
                for lane in 0..4 {
                    let byte = offset + lane * 4;
                    let value = i32::from_le_bytes(
                        ctx.banks[physical][byte..byte + 4].try_into().unwrap(),
                    );
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
