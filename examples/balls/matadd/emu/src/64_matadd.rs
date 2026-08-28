use super::super::bank::{bank_num, bank_size};
use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{BallInstruction, ExecContext};

pub struct MatAdd;

impl BallInstruction for MatAdd {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let a_bank = rs1_b0(xs1);
        let b_bank = rs1_b1(xs1);
        let c_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;

        if xs2 != 0 {
            panic!("matadd: rs2 must be zero");
        }
        if a_bank >= bank_num() as u64 || b_bank >= bank_num() as u64 || c_bank >= bank_num() as u64 {
            panic!("matadd: invalid bank id");
        }
        if a_bank == b_bank || a_bank == c_bank || b_bank == c_bank {
            panic!("matadd: banks must be distinct");
        }
        if iter == 0 || iter * 16 > bank_size() {
            panic!("matadd: iter must fit in one physical bank");
        }

        let a = &ctx.cfgs[a_bank as usize];
        let b = &ctx.cfgs[b_bank as usize];
        let c = &ctx.cfgs[c_bank as usize];
        if !a.allocated || !b.allocated || !c.allocated {
            panic!("matadd: all banks must be allocated");
        }
        if a.cols == 0 || a.cols != b.cols || a.cols != c.cols {
            panic!("matadd: bank groups must match");
        }

        for group in 0..a.cols as usize {
            let a_pbank = pbank_group(ctx.bank_map, a_bank, group as u64);
            let b_pbank = pbank_group(ctx.bank_map, b_bank, group as u64);
            let c_pbank = pbank_group(ctx.bank_map, c_bank, group as u64);
            for line in 0..iter {
                let offset = line * 16;
                for lane in 0..4 {
                    let byte = offset + lane * 4;
                    let a_value = u32::from_le_bytes(ctx.banks[a_pbank][byte..byte + 4].try_into().unwrap());
                    let b_value = u32::from_le_bytes(ctx.banks[b_pbank][byte..byte + 4].try_into().unwrap());
                    ctx.banks[c_pbank][byte..byte + 4]
                        .copy_from_slice(&a_value.wrapping_add(b_value).to_le_bytes());
                }
            }
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        if xs2 != 0 {
            panic!("matadd: rs2 must be zero");
        }
        let iter = rs1_iter(xs1);
        if iter == 0 || iter * 16 > bank_size() as u64 {
            panic!("matadd: iter must fit in one physical bank");
        }
        iter * 4
    }
}
