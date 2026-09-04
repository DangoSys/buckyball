use super::super::bank::{bank_lines, bank_num, bank_row_bytes};
use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub struct Int8Mul;

impl Int8Mul {
    fn fields(xs2: u64) -> (f32, usize) {
        if xs2 >> 38 != 0 {
            panic!("int8mul: reserved rs2 bits must be zero");
        }
        let ratio = f32::from_bits(xs2 as u32);
        let gate_row = ((xs2 >> 32) & 0x3f) as usize;
        if !ratio.is_finite() || ratio <= 0.0 {
            panic!("int8mul: ratio must be finite and positive");
        }
        if gate_row >= bank_lines() {
            panic!("int8mul: gate row must fit one physical bank");
        }
        (ratio, gate_row)
    }

    pub fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let gate_bank = rs1_b0(xs1);
        let input_bank = rs1_b1(xs1);
        let output_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;
        let (ratio, gate_row) = Self::fields(xs2);

        if gate_bank >= bank_num() as u64
            || input_bank >= bank_num() as u64
            || output_bank >= bank_num() as u64
        {
            panic!("int8mul: invalid bank id");
        }
        if gate_bank == input_bank || gate_bank == output_bank || input_bank == output_bank {
            panic!("int8mul: banks must be distinct");
        }
        if iter == 0 || iter > bank_lines() {
            panic!("int8mul: iter must fit one physical bank");
        }
        let gate = &ctx.cfgs[gate_bank as usize];
        let input = &ctx.cfgs[input_bank as usize];
        let output = &ctx.cfgs[output_bank as usize];
        if !gate.allocated || !input.allocated || !output.allocated {
            panic!("int8mul: all banks must be allocated");
        }
        if gate.cols == 0 || gate.cols != input.cols || gate.cols != output.cols {
            panic!("int8mul: bank groups must match");
        }

        for group in 0..gate.cols as usize {
            let pg = pbank_group(ctx.bank_map, gate_bank, group as u64);
            let pi = pbank_group(ctx.bank_map, input_bank, group as u64);
            let po = pbank_group(ctx.bank_map, output_bank, group as u64);
            let gate_base = gate_row * bank_row_bytes();
            let mut gate_values = [0u8; 16];
            gate_values.copy_from_slice(&ctx.banks[pg][gate_base..gate_base + 16]);
            for row in 0..iter {
                let base = row * bank_row_bytes();
                let mut input_values = [0u8; 16];
                input_values.copy_from_slice(&ctx.banks[pi][base..base + 16]);
                for lane in 0..16 {
                    let value = (gate_values[lane] as i8 as f32)
                        * (input_values[lane] as i8 as f32)
                        * ratio;
                    ctx.banks[po][base + lane] =
                        value.round_ties_even().clamp(-128.0, 127.0) as i8 as u8;
                }
            }
        }
        0
    }

    pub fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        Self::fields(xs2);
        if iter == 0 || iter > bank_lines() as u64 {
            panic!("int8mul: iter must fit one physical bank");
        }
        2 + iter * 20
    }
}
