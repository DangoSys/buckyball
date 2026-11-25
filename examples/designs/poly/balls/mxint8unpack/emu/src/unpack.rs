use bebop_bemu::ball::blink::bank_lines;
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b2, rs1_iter};
use bebop_bemu::ExecContext;

fn shape(a: u64, b: u64) -> usize {
    let blocks = rs1_iter(a) as usize;
    assert!(b == 0, "mxint8unpack: reserved encoding must be zero");
    assert!((1..=128).contains(&blocks), "mxint8unpack: requires 1..128 blocks");
    blocks
}

fn bits(code: i8, exponent: u8) -> u32 {
    if code == 0 {
        return 0;
    }
    let magnitude = i32::from(code).unsigned_abs();
    let highest = 31 - magnitude.leading_zeros();
    let sign = if code < 0 { 0x80000000 } else { 0 };
    let field = u32::from(exponent) + highest;
    if field >= 255 {
        sign | 0x7f800000
    } else if field == 0 {
        sign | 0x00400000
    } else {
        sign | (field << 23) | ((magnitude << (23 - highest)) & 0x7fffff)
    }
}

pub fn execute(a: u64, b: u64, ctx: &mut ExecContext) -> u64 {
    let blocks = shape(a, b);
    let input = rs1_b0(a);
    let output = rs1_b2(a);
    assert!(input != output, "mxint8unpack: banks must be distinct");
    assert!(bank_lines() == 1024, "mxint8unpack: requires 16 KiB banks");
    for bank in [input, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxint8unpack: requires allocated single-group banks"
        );
    }
    let source = pbank(ctx, input);
    let destination = pbank(ctx, output);
    for block in 0..blocks {
        let exponent = ctx.banks[source][32 * blocks + block];
        for lane in 0..32 {
            let at = 32 * block + lane;
            let value = bits(ctx.banks[source][at] as i8, exponent);
            ctx.banks[destination][4 * at..4 * at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    ctx.config_mut(output).valid_rows = (8 * blocks) as u64;
    0
}

pub fn cycles(a: u64, b: u64) -> u64 {
    super::latency::cycles(shape(a, b) as u64)
}

#[cfg(test)]
mod tests {
    #[test]
    fn bank_latency_matches_rtl_shapes() {
        for (blocks, cycles) in [(1, 22), (8, 155), (17, 328), (128, 2449)] {
            assert_eq!(super::cycles(blocks << 30, 0), cycles);
        }
    }

    #[test]
    fn all_codes_and_exponents_match_exact_binary_scaling() {
        for exponent in 0..=255 {
            for code in -128..=127 {
                let reference = (f64::from(code) * 2.0_f64.powi(exponent - 127)) as f32;
                assert_eq!(super::bits(code as i8, exponent as u8), reference.to_bits());
            }
        }
    }

    #[test]
    #[should_panic(expected = "1..128 blocks")]
    fn zero_blocks_fail() {
        super::shape(0, 0);
    }

    #[test]
    #[should_panic(expected = "1..128 blocks")]
    fn oversized_output_fails() {
        super::shape(129 << 30, 0);
    }

    #[test]
    #[should_panic(expected = "reserved encoding")]
    fn reserved_rs2_fails() {
        super::shape(1 << 30, 1);
    }

    #[test]
    fn unused_bank_is_not_an_operand() {
        assert_eq!(super::shape((1 << 30) | (7 << 10), 0), 1);
    }
}
