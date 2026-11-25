use bebop_bemu::ball::blink::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use bebop_bemu::ball::ExecContext;

fn rounded(value: i64, divisor: u64) -> i64 {
    let n = value.unsigned_abs();
    let q = n / divisor;
    let r = n % divisor;
    let result = q + u64::from(r * 2 > divisor || (r * 2 == divisor && q & 1 != 0));
    if value < 0 {
        (-(result as i128)) as i64
    } else {
        result as i64
    }
}

fn word(ctx: &ExecContext, bank: usize, index: usize) -> i32 {
    i32::from_le_bytes(ctx.banks[bank][index * 4..index * 4 + 4].try_into().unwrap())
}

fn calculate(a: i32, b: i32, multiplier: i32, shift: u32, mode: u64, fraction: u32) -> i32 {
    let unit = 1i64 << fraction;
    let q = if mode <= 2 {
        assert!(
            (0..=1 << 30).contains(&multiplier) && shift <= 61,
            "fixed: illegal coefficient"
        );
        rounded(i64::from(a) * i64::from(multiplier), 1u64 << shift).clamp(i32::MIN as i64, i32::MAX as i64)
    } else {
        i64::from(a)
    };
    let result = match mode {
        0 => q,
        1 if q <= -3 * unit => 0,
        1 if q >= 3 * unit => q,
        1 if q == 0 => 0,
        1 => rounded(q * (q + 3 * unit), (6 * unit) as u64),
        2 if q <= -3 * unit => 0,
        2 if q >= 3 * unit => unit,
        2 if q == 0 => unit / 2,
        2 => rounded(q + 3 * unit, 6),
        3 => rounded(i64::from(a) * i64::from(b), unit as u64),
        4 => q.max(0),
        5 => q + i64::from(b),
        _ => panic!("fixed: illegal mode"),
    };
    result.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn banks(xs1: u64, xs2: u64, ctx: &ExecContext) -> ([usize; 3], usize, u64, u32, bool, bool) {
    let ids = [rs1_b0(xs1), rs1_b1(xs1), rs1_b2(xs1)];
    let rows = rs1_iter(xs1) as usize;
    let mode = xs2 & 7;
    let fraction = ((xs2 >> 4) & 31) as u32;
    let broadcast = xs2 & 8 != 0;
    let stream = xs2 & 512 != 0;
    assert!(
        xs1 >> 63 == 0 && xs1 & (1 << 29) == 0 && xs2 >> 10 == 0,
        "fixed: reserved bits"
    );
    assert!(
        mode <= 7 && (1..=24).contains(&fraction),
        "fixed: illegal mode or fraction"
    );
    assert!(!stream || mode <= 2, "fixed: only requantization streams coefficients");
    assert!(!broadcast || mode == 3, "fixed: only multiply broadcasts");
    assert!(mode != 7 || rows == 4, "fixed: mean requires four output rows");
    assert!(
        rows > 0 && rows % 4 == 0,
        "fixed: rows must be a positive multiple of four"
    );
    assert!(
        ids[0] != ids[1] && ids[0] != ids[2] && (ids[1] != ids[2] || mode == 6),
        "fixed: banks must differ except sum may write B"
    );
    let mut physical = [0; 3];
    for (i, id) in ids.into_iter().enumerate() {
        assert!(
            !bebop_bemu::ball::blink::is_shared_vbank(id),
            "fixed: private banks required"
        );
        assert!(
            ctx.config(id).allocated && ctx.config(id).cols == 1,
            "fixed: one allocated group required"
        );
        physical[i] = pbank_group(ctx, id, 0);
        let needed = if (mode == 6 && i != 0) || (mode == 7 && i == 0) {
            8
        } else if i == 1 && mode <= 2 {
            if stream {
                rows * 2
            } else {
                8
            }
        } else if i == 1 && broadcast {
            4
        } else {
            rows
        };
        assert!(needed <= ctx.banks[physical[i]].len() / 16, "fixed: bank span exceeded");
    }
    (physical, rows, mode, fraction, broadcast, stream)
}

pub fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let ([a, b, c], rows, mode, fraction, broadcast, stream) = banks(xs1, xs2, ctx);
    if mode >= 6 {
        for channel in 0..16 {
            let source = if mode == 6 { b } else { a };
            let mut sum = i64::from_le_bytes(ctx.banks[source][channel * 8..channel * 8 + 8].try_into().unwrap());
            if mode == 6 {
                for pixel in 0..rows / 4 {
                    sum = sum
                        .checked_add(i64::from(word(ctx, a, pixel * 16 + channel)))
                        .expect("fixed: sum overflow");
                }
                ctx.banks[c][channel * 8..channel * 8 + 8].copy_from_slice(&sum.to_le_bytes());
            } else {
                let divisor = word(ctx, b, channel);
                assert!(divisor > 0, "fixed: mean requires positive divisors");
                let value = rounded(sum, divisor as u64).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                ctx.banks[c][channel * 4..channel * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        ctx.config_mut(rs1_b2(xs1)).valid_rows = if mode == 6 { 8 } else { 4 };
        return 0;
    }
    for i in 0..rows * 4 {
        let right = if mode == 3 || mode == 5 {
            word(ctx, b, if broadcast { i % 16 } else { i })
        } else {
            0
        };
        let (multiplier, shift) = if mode <= 2 {
            {
                let m = if stream { i / 4 * 8 + i % 4 } else { i % 16 };
                let shift = if stream { m + 4 } else { m + 16 };
                (word(ctx, b, m), word(ctx, b, shift) as u32)
            }
        } else {
            (0, 0)
        };
        let value = calculate(word(ctx, a, i), right, multiplier, shift, mode, fraction);
        ctx.banks[c][i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    ctx.config_mut(rs1_b2(xs1)).valid_rows = rows as u64;
    0
}

pub fn cycles(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let ([a, b, _], rows, _, _, _, _) = banks(xs1, xs2, ctx);
    let words = |bank: usize| -> Vec<i32> { (0..ctx.banks[bank].len() / 4).map(|i| word(ctx, bank, i)).collect() };
    super::latency::cycles(rows as u64, xs2, &words(a), &words(b))
}
