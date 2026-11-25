//! MXINT8ACT reference, ported from the BEMU model (emu/src/activation.rs),
//! with counters of the semantic paths each case exercises.

// Lookup paths.
pub const TINY: usize = 0;
pub const HIGH_TAIL: usize = 1;
pub const LOW_TAIL: usize = 2;
pub const INTERIOR: usize = 3;
pub const WORD_CROSS: usize = 4;
pub const WIDE_ZERO: usize = 5;
// Rescale of the interpolated value by exponent + 10.
pub const RESCALE_RIGHT: usize = 6;
pub const RESCALE_EXACT: usize = 7;
pub const RESCALE_LEFT: usize = 8;
// Output block exponent.
pub const ZERO_BLOCK: usize = 9;
pub const CLAMP_LOW: usize = 10;
pub const CLAMP_HIGH: usize = 11;
// Output code encoding.
pub const ROUND_TIE: usize = 12;
pub const CODE_CLAMP: usize = 13;
pub const SCALE_SATURATE: usize = 14;
pub const ENCODE_LEFT: usize = 15;
pub const ENCODE_FLUSH: usize = 16;
// Gate codes.
pub const GATE_ZERO: usize = 17;
pub const GATE_NEGATIVE: usize = 18;
pub const EVENTS: usize = 19;

pub fn row_stride(k: usize) -> usize {
    (k + k / 32).div_ceil(16)
}

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

fn lookup(code: i8, exponent: i32, table: &[i32], events: &mut [u32; EVENTS]) -> i64 {
    assert!(code >= -127, "code -128 is reserved");
    if exponent < -24 {
        events[TINY] += 1;
        return i64::from(code) << 15;
    }
    let x = i64::from(code) << (exponent.min(4) + 26);
    if x >= 16i64 << 26 {
        events[HIGH_TAIL] += 1;
        return i64::from(code) << 16;
    }
    if x <= -(16i64 << 26) {
        events[LOW_TAIL] += 1;
        return 0;
    }
    let position = x + (16i64 << 26);
    let index = (position >> 19) as usize;
    events[INTERIOR] += 1;
    // Index 4095 needs x >= 2^30 - 2^19, which no legal code reaches below the high tail.
    if index & 3 == 3 {
        events[WORD_CROSS] += 1;
    }
    if exponent > 4 {
        events[WIDE_ZERO] += 1;
    }
    let shift = exponent + 10;
    events[if shift > 0 {
        RESCALE_RIGHT
    } else if shift == 0 {
        RESCALE_EXACT
    } else {
        RESCALE_LEFT
    }] += 1;
    let first = i64::from(table[index]);
    let last = if index == 4095 {
        16i64 << 26
    } else {
        i64::from(table[index + 1])
    };
    let value = first + round_shift((last - first) * (position & ((1 << 19) - 1)), 19);
    round_shift(value, shift)
}

/// Writes the output span of `output` (bank bytes) as the Ball does.
/// Bases are in 16-byte bank rows.
#[allow(clippy::too_many_arguments)]
pub fn activate(
    input: &[u8],
    table_bank: &[u8],
    output: &mut [u8],
    rows: usize,
    k: usize,
    gated: bool,
    in_base: usize,
    out_base: usize,
    table_base: usize,
    events: &mut [u32; EVENTS],
) {
    let stride = row_stride(k) * 16;
    let table = table_bank[table_base * 16..table_base * 16 + 16384]
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    output[out_base * 16..out_base * 16 + rows * stride].fill(0);
    for row in 0..rows {
        let source = in_base * 16 + row * stride;
        let gate = source + rows * stride;
        let target = out_base * 16 + row * stride;
        for block in 0..k / 32 {
            assert!(input[source + k + block] != 255, "exponent byte 255 is illegal");
            let exponent = i32::from(input[source + k + block]) - 127;
            let gate_exponent = if gated {
                assert!(input[gate + k + block] != 255, "gate exponent byte 255 is illegal");
                i32::from(input[gate + k + block]) - 127
            } else {
                0
            };
            let scale = exponent + gate_exponent;
            let mut values = [0i64; 32];
            for (lane, value) in values.iter_mut().enumerate() {
                *value = lookup(input[source + block * 32 + lane] as i8, exponent, &table, events);
                if gated {
                    let code = input[gate + block * 32 + lane] as i8;
                    assert!(code >= -127, "gate code -128 is reserved");
                    if code == 0 {
                        events[GATE_ZERO] += 1;
                    }
                    if code < 0 {
                        events[GATE_NEGATIVE] += 1;
                    }
                    *value *= i64::from(code);
                }
            }
            let maximum = values.iter().map(|value| value.unsigned_abs()).max().unwrap();
            let candidate = if maximum == 0 {
                0
            } else {
                scale + maximum.ilog2() as i32 - 22
            };
            if maximum == 0 {
                events[ZERO_BLOCK] += 1;
            } else if candidate < -127 {
                events[CLAMP_LOW] += 1;
            } else if candidate > 127 {
                events[CLAMP_HIGH] += 1;
            }
            let output_exponent = candidate.clamp(-127, 127);
            let shift = output_exponent - scale + 16;
            for (lane, value) in values.iter().enumerate() {
                let code = if shift < -6 {
                    if *value != 0 {
                        events[SCALE_SATURATE] += 1;
                    }
                    value.signum() * 127
                } else {
                    if *value != 0 && shift < 0 {
                        events[ENCODE_LEFT] += 1;
                    }
                    if *value != 0 && shift > 62 {
                        events[ENCODE_FLUSH] += 1;
                    }
                    if (1..=62).contains(&shift) && value.unsigned_abs() & ((1u64 << shift) - 1) == 1u64 << (shift - 1)
                    {
                        events[ROUND_TIE] += 1;
                    }
                    let rounded = round_shift(*value, shift);
                    if rounded.abs() > 127 {
                        events[CODE_CLAMP] += 1;
                    }
                    rounded.clamp(-127, 127)
                };
                output[target + block * 32 + lane] = code as i8 as u8;
            }
            output[target + k + block] = (output_exponent + 127) as u8;
        }
    }
}
