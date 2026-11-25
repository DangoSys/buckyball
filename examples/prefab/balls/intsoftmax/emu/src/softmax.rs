use bebop_bemu::ball::blink::bank_lines;
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use bebop_bemu::ExecContext;

fn q8(bits: u32) -> i32 {
    let exponent = (bits >> 23) & 255;
    assert!(exponent != 255 || bits & 0x7fffff == 0, "intsoftmax: NaN score");
    let magnitude = if exponent >= 139 {
        1 << 20
    } else if exponent < 118 {
        0
    } else {
        let mantissa = (bits & 0x7fffff) | 0x800000;
        let shift = 142 - exponent;
        let quotient = mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        quotient + u32::from(remainder > half || (remainder == half && quotient & 1 != 0))
    };
    if bits >> 31 != 0 {
        -(magnitude as i32)
    } else {
        magnitude as i32
    }
}

fn f32_q15(value: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    let highest = 31 - value.leading_zeros();
    ((highest + 112) << 23) | ((value << (23 - highest)) & 0x7fffff)
}

fn shape(a: u64, b: u64) -> (usize, usize, usize) {
    let rows = rs1_iter(a) as usize;
    let columns = b as usize & 255;
    assert!(
        b >> 8 == 0 && (1..=128).contains(&columns),
        "intsoftmax: illegal width or encoding"
    );
    let stride = columns.div_ceil(4);
    assert!(
        rows > 0 && rows * stride <= bank_lines(),
        "intsoftmax: rows exceed bank"
    );
    (rows, columns, stride)
}

pub fn execute(a: u64, b: u64, ctx: &mut ExecContext) -> u64 {
    let (rows, columns, stride) = shape(a, b);
    let banks = [rs1_b0(a), rs1_b1(a), rs1_b2(a)];
    assert!(
        banks[0] != banks[1] && banks[0] != banks[2] && banks[1] != banks[2],
        "intsoftmax: banks must be distinct"
    );
    for bank in banks {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "intsoftmax: requires allocated single-group banks"
        );
    }
    assert!(bank_lines() == 1024, "intsoftmax: requires 16 KiB banks");
    let [input, table, output] = banks.map(|bank| pbank(ctx, bank));
    let table: Vec<u32> = ctx.banks[table][..16384]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    for row in 0..rows {
        let begin = row * stride * 16;
        let scores: Vec<i32> = ctx.banks[input][begin..begin + columns * 4]
            .chunks_exact(4)
            .map(|word| q8(u32::from_le_bytes(word.try_into().unwrap())))
            .collect();
        let maximum = *scores.iter().max().unwrap();
        let weights: Vec<u32> = scores
            .iter()
            .map(|&value| {
                let index = (maximum - value).min(4095) as usize;
                let entry = table[index];
                assert!(
                    entry <= 32768 && (index != 0 || entry == 32768),
                    "intsoftmax: illegal accessed Q15 table entry"
                );
                entry
            })
            .collect();
        let sum: u64 = weights.iter().map(|&value| u64::from(value)).sum();
        ctx.banks[output][begin..begin + stride * 16].fill(0);
        for (column, value) in weights.into_iter().enumerate() {
            let probability = ((u64::from(value) * 32768 + sum / 2) / sum) as u32;
            let at = begin + column * 4;
            ctx.banks[output][at..at + 4].copy_from_slice(&f32_q15(probability).to_le_bytes());
        }
    }
    ctx.config_mut(banks[2]).valid_rows = (rows * stride) as u64;
    0
}

pub fn cycles(a: u64, b: u64) -> u64 {
    let (rows, _, _) = shape(a, b);
    super::latency::cycles(rows as u64, b)
}
