use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use bebop_bemu::ExecContext;

fn round_shift(value: i64, shift: i32) -> i64 {
    if shift <= 0 {
        return value << -shift;
    }
    if shift > 62 {
        return 0;
    }
    let magnitude = value.unsigned_abs();
    let quotient = magnitude >> shift;
    let remainder = magnitude & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let rounded = quotient + u64::from(remainder > half || (remainder == half && quotient & 1 != 0));
    if value < 0 {
        -(rounded as i64)
    } else {
        rounded as i64
    }
}

fn lookup(code: i8, exponent: i32, table: &[i32]) -> i64 {
    assert!(code >= -127, "MXINT8ACT code -128 is reserved");
    if exponent < -24 {
        return i64::from(code) << 15;
    }
    let x = i64::from(code) << (exponent.min(4) + 26);
    if x >= 16i64 << 26 {
        return i64::from(code) << 16;
    }
    if x <= -(16i64 << 26) {
        return 0;
    }
    let position = x + (16i64 << 26);
    let index = (position >> 19) as usize;
    let first = i64::from(table[index]);
    let last = if index == 4095 {
        16i64 << 26
    } else {
        i64::from(table[index + 1])
    };
    let value = first + round_shift((last - first) * (position & ((1 << 19) - 1)), 19);
    round_shift(value, exponent + 10)
}

pub(crate) fn execute(rs1: u64, rs2: u64, ctx: &mut ExecContext) -> u64 {
    assert!(
        rs2 >> 59 == 0 && !rs1_read_old(rs1) && !rs1_update(rs1),
        "MXINT8ACT reserved fields must be zero"
    );
    let banks = [rs1_b0(rs1), rs1_b1(rs1), rs1_b2(rs1)];
    assert!(
        banks[0] != banks[1] && banks[0] != banks[2] && banks[1] != banks[2],
        "MXINT8ACT banks must differ"
    );
    for bank in banks {
        let config = ctx.config(bank);
        assert!(config.allocated && config.cols == 1, "MXINT8ACT needs one-group banks");
    }
    let input = pbank(ctx, banks[0]);
    let table_bank = pbank(ctx, banks[1]);
    let output = pbank(ctx, banks[2]);
    let rows = rs1_iter(rs1) as usize;
    let k = ((rs2 >> 1) & 65535) as usize;
    let gated = rs2 & 1 != 0;
    assert!(
        rows > 0 && k > 0 && k % 32 == 0,
        "MXINT8ACT needs rows and a multiple of 32 values"
    );
    let input_k = k;
    let input_stride = (input_k + input_k / 32).div_ceil(16) * 16;
    let output_stride = (k + k / 32).div_ceil(16) * 16;
    let input_base = ((rs2 >> 17) & 16383) as usize * 16;
    let output_base = ((rs2 >> 31) & 16383) as usize * 16;
    let table_base = ((rs2 >> 45) & 16383) as usize * 16;
    assert!(
        input_base + rows * input_stride * if gated { 2 } else { 1 } <= ctx.banks[input].len(),
        "MXINT8ACT input exceeds bank"
    );
    assert!(
        output_base + rows * output_stride <= ctx.banks[output].len(),
        "MXINT8ACT output exceeds bank"
    );
    assert!(
        table_base + 16384 <= ctx.banks[table_bank].len(),
        "MXINT8ACT table exceeds bank"
    );
    let table = ctx.banks[table_bank][table_base..table_base + 16384]
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    ctx.banks[output][output_base..output_base + rows * output_stride].fill(0);
    for row in 0..rows {
        let source = input_base + row * input_stride;
        let target = output_base + row * output_stride;
        for block in 0..k / 32 {
            let exponent = i32::from(ctx.banks[input][source + input_k + block]) - 127;
            assert!(exponent < 128, "MXINT8ACT requires finite MXINT8 exponents");
            let gate_exponent = if gated {
                let exponent = i32::from(ctx.banks[input][source + rows * input_stride + k + block]) - 127;
                assert!(exponent < 128, "MXINT8ACT gate exponent must be finite");
                exponent
            } else {
                0
            };
            let scale = exponent + gate_exponent;
            let mut values = [0i64; 32];
            for (lane, value) in values.iter_mut().enumerate() {
                *value = lookup(ctx.banks[input][source + block * 32 + lane] as i8, exponent, &table);
                if gated {
                    let code = ctx.banks[input][source + rows * input_stride + block * 32 + lane] as i8;
                    assert!(code >= -127, "MXINT8ACT gate code -128 is reserved");
                    *value *= i64::from(code);
                }
            }
            let maximum = values.iter().map(|value| value.unsigned_abs()).max().unwrap();
            let output_exponent = if maximum == 0 {
                0
            } else {
                (scale + maximum.ilog2() as i32 - 22).clamp(-127, 127)
            };
            for (lane, value) in values.iter().enumerate() {
                let shift = output_exponent - scale + 16;
                let code = if shift < -6 {
                    value.signum() * 127
                } else {
                    round_shift(*value, shift).clamp(-127, 127)
                };
                ctx.banks[output][target + block * 32 + lane] = code as i8 as u8;
            }
            ctx.banks[output][target + k + block] = (output_exponent + 127) as u8;
        }
    }
    0
}

pub(crate) fn cycles(rs1: u64, rs2: u64, ctx: &ExecContext) -> u64 {
    super::latency::cycles(rs1_iter(rs1), rs2, &ctx.banks[pbank(ctx, rs1_b0(rs1))])
}
