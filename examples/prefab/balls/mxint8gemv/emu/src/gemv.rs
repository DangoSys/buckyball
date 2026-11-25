use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use bebop_bemu::ExecContext;

fn round_shift(value: i64, shift: u32) -> i64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }
    let magnitude = value.unsigned_abs();
    let quotient = magnitude >> shift;
    let remainder = magnitude & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    let rounded = quotient + u64::from(remainder > halfway || remainder == halfway && quotient & 1 != 0);
    if value < 0 {
        -(rounded as i64)
    } else {
        rounded as i64
    }
}

pub(crate) fn execute(rs1: u64, rs2: u64, ctx: &mut ExecContext) -> u64 {
    let (a, b, c) = (rs1_b0(rs1), rs1_b1(rs1), rs1_b2(rs1));
    let k = rs1_iter(rs1) as usize;
    let a_codes = (rs2 & 0x3fff) as usize;
    let a_exponents = ((rs2 >> 14) & 0x3fff) as usize;
    let b_base = ((rs2 >> 28) & 0x3fff) as usize;
    let c_base = ((rs2 >> 42) & 0x3fff) as usize;
    let accumulate = rs2 & (1 << 56) != 0;
    let finalize = rs2 & (1 << 57) != 0;
    assert!(
        rs2 >> 58 == 0 && !rs1_update(rs1),
        "mxint8gemv: reserved fields must be zero"
    );
    assert_eq!(
        rs1_read_old(rs1),
        accumulate,
        "mxint8gemv: inconsistent accumulate fields"
    );
    assert!(k > 0 && k % 32 == 0, "mxint8gemv: K must be positive and aligned to 32");
    assert!(a != b && a != c && b != c, "mxint8gemv: banks must differ");
    for (bank, groups) in [(a, 1), (b, 4), (c, 1)] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == groups,
            "mxint8gemv: invalid bank group allocation"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "mxint8gemv: bank rows must be 128 bits");
    let blocks = k / 32;
    let stride = k / 16 + k.div_ceil(512);
    assert!(
        a_codes + k / 16 <= bank_lines()
            && a_exponents + blocks.div_ceil(16) <= bank_lines()
            && b_base + 8 * stride <= bank_lines()
            && c_base + 17 <= bank_lines(),
        "mxint8gemv: bank window exceeds capacity"
    );
    let pa = ctx.physical_bank(a, 0);
    let pc = ctx.physical_bank(c, 0);
    let codes = ctx.banks[pa][a_codes * 16..a_codes * 16 + k].to_vec();
    let exponents = ctx.banks[pa][a_exponents * 16..a_exponents * 16 + blocks].to_vec();
    let weights: Vec<Vec<u8>> = (0..4)
        .map(|group| {
            let physical = ctx.physical_bank(b, group);
            ctx.banks[physical][b_base * 16..(b_base + 8 * stride) * 16].to_vec()
        })
        .collect();
    let mut values = [0i64; 32];
    let mut scale = 0i16;
    let mut valid = false;
    let mut processed = 0usize;
    if accumulate {
        let state = &ctx.banks[pc][c_base * 16..c_base * 16 + 272];
        processed = u16::from_le_bytes(state[0..2].try_into().unwrap()) as usize;
        scale = i16::from_le_bytes(state[2..4].try_into().unwrap());
        assert!(
            state[4] <= 1 && state[5..16].iter().all(|x| *x == 0),
            "mxint8gemv: invalid state header"
        );
        valid = state[4] != 0;
        for (j, value) in values.iter_mut().enumerate() {
            *value = i64::from_le_bytes(state[16 + j * 8..24 + j * 8].try_into().unwrap());
        }
        assert!(
            processed > 0 && processed <= 1024,
            "mxint8gemv: invalid state block count"
        );
        assert!(
            if valid {
                (-254..=254).contains(&scale) && values.iter().any(|x| *x != 0)
            } else {
                scale == 0 && values.iter().all(|x| *x == 0)
            },
            "mxint8gemv: invalid state scale"
        );
        assert!(
            values.iter().all(|x| x.unsigned_abs() <= (processed as u64) << 51),
            "mxint8gemv: invalid state magnitude"
        );
    }
    assert!(processed + blocks <= 1024, "mxint8gemv: total K exceeds 32768");
    for block in 0..blocks {
        let mut products = [0i64; 32];
        let mut scales = [0i16; 32];
        assert!(exponents[block] != 255, "mxint8gemv: reserved A exponent");
        for j in 0..32 {
            let row = &weights[j / 8][j % 8 * stride * 16..(j % 8 + 1) * stride * 16];
            assert!(row[k + block] != 255, "mxint8gemv: reserved B exponent");
            scales[j] = exponents[block] as i16 + row[k + block] as i16 - 254;
            products[j] = (0..32)
                .map(|i| i64::from(codes[block * 32 + i] as i8) * i64::from(row[block * 32 + i] as i8))
                .sum();
        }
        if let Some(incoming) = (0..32).filter(|j| products[*j] != 0).map(|j| scales[j]).max() {
            let next = if valid { scale.max(incoming) } else { incoming };
            if valid && next > scale {
                for value in &mut values {
                    *value = round_shift(*value, (next - scale) as u32);
                }
            }
            for j in 0..32 {
                if products[j] == 0 {
                    continue;
                }
                let shift = 32 + i32::from(scales[j]) - i32::from(next);
                let term = if shift >= 0 {
                    products[j] << shift
                } else {
                    round_shift(products[j], (-shift) as u32)
                };
                values[j] = values[j].checked_add(term).expect("mxint8gemv: INT64 overflow");
            }
            valid = values.iter().any(|x| *x != 0);
            scale = if valid { next } else { 0 };
        }
    }
    processed += blocks;
    if finalize {
        let maximum = values.iter().map(|x| x.unsigned_abs()).max().unwrap();
        let exponent = if maximum == 0 {
            0
        } else {
            (63 - maximum.leading_zeros() as i32 + i32::from(scale) - 32 - 6).clamp(-127, 127)
        };
        let shift = exponent - i32::from(scale) + 32;
        let mut output = [0u8; 48];
        for j in 0..32 {
            let code = if shift >= 0 {
                round_shift(values[j], shift as u32)
            } else if shift < -6 || values[j].unsigned_abs() > (127u64 >> -shift) {
                values[j].signum() * 127
            } else {
                values[j] << -shift
            };
            output[j] = code.clamp(-127, 127) as i8 as u8;
        }
        output[32] = (exponent + 127) as u8;
        ctx.banks[pc][c_base * 16..c_base * 16 + 48].copy_from_slice(&output);
    } else {
        let mut output = [0u8; 272];
        output[0..2].copy_from_slice(&(processed as u16).to_le_bytes());
        output[2..4].copy_from_slice(&scale.to_le_bytes());
        output[4] = u8::from(valid);
        for (j, value) in values.iter().enumerate() {
            output[16 + j * 8..24 + j * 8].copy_from_slice(&value.to_le_bytes());
        }
        ctx.banks[pc][c_base * 16..c_base * 16 + 272].copy_from_slice(&output);
    }
    if bebop_bemu::ball::blink::mtrace_enabled() {
        for row in 0..if finalize { 3 } else { 17 } {
            let bytes = &ctx.banks[pc][(c_base + row) * 16..(c_base + row + 1) * 16];
            bebop_bemu::ball::blink::mtrace(bebop_bemu::ball::blink::MTraceEvent {
                is_write: true,
                is_shared: bebop_bemu::ball::blink::is_shared_vbank(c),
                channel: 0,
                hart_id: ctx.hart_id as u64,
                rob_id: ctx.inst_id as u32,
                vbank_id: c as u32,
                pbank_id: ctx.reported_physical_bank(c, pc),
                group_id: 0,
                addr: (c_base + row) as u32,
                write_mask: 0xffff,
                data_lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                data_hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            });
        }
    }
    0
}

// The cycles of the Ball against an ideal outside world (latency.rs), from the
// A and B banks the instruction reads.
pub(crate) fn cycles(rs1: u64, rs2: u64, ctx: &ExecContext) -> u64 {
    let k = rs1_iter(rs1);
    assert!(k > 0, "mxint8gemv: K must be positive");
    let a = &ctx.banks[ctx.physical_bank(rs1_b0(rs1), 0)];
    let b = [0u64, 1, 2, 3].map(|group| &ctx.banks[ctx.physical_bank(rs1_b1(rs1), group)][..]);
    super::latency::cycles(k, rs2, a, b)
}
