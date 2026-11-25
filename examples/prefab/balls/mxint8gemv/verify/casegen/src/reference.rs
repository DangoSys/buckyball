// Reference of one MXINT8GEMV instruction, ported from emu/src/gemv.rs and
// CONTRACT.md. It also records which numerical paths a case exercised.

#[derive(Default)]
pub struct Flags {
    pub align: bool,
    pub align_flush: bool,
    pub term_round: bool,
    pub term_flush: bool,
    pub tie: bool,
    pub zero_block: bool,
    pub restart: bool,
    pub q_high: bool,
    pub q_low: bool,
    pub code_sat: bool,
    pub round_clamp: bool,
    pub left_shift: bool,
    pub zero_out: bool,
    pub edge_code: bool,
}

pub struct State {
    pub values: [i64; 32],
    pub scale: i16,
    pub valid: bool,
    pub processed: usize,
}

pub struct Command {
    pub k: usize,
    pub a_codes: usize,
    pub a_exp: usize,
    pub b_base: usize,
    pub c_base: usize,
    pub accumulate: bool,
    pub finalize: bool,
}

pub fn stride(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

pub fn round_shift(value: i64, shift: u32, flags: &mut Flags) -> i64 {
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
    flags.tie |= remainder == halfway;
    let rounded = quotient + u64::from(remainder > halfway || remainder == halfway && quotient & 1 != 0);
    if value < 0 {
        -(rounded as i64)
    } else {
        rounded as i64
    }
}

// One K/32 block: exact INT32 dots, then the shared-scale accumulation.
pub fn block(state: &mut State, a: &[i8], a_exp: u8, b: &[&[i8]], b_exp: &[u8], flags: &mut Flags) {
    assert!(a_exp != 255 && b_exp.iter().all(|e| *e != 255), "reserved exponent");
    let mut products = [0i64; 32];
    let mut scales = [0i16; 32];
    for j in 0..32 {
        scales[j] = a_exp as i16 + b_exp[j] as i16 - 254;
        for i in 0..32 {
            products[j] += i64::from(a[i]) * i64::from(b[j][i]);
            flags.edge_code |= (a[i] == -128 && b[j][i] != 0) || (b[j][i] == -128 && a[i] != 0);
        }
    }
    let incoming = (0..32).filter(|j| products[*j] != 0).map(|j| scales[j]).max();
    let Some(incoming) = incoming else {
        flags.zero_block = true;
        return;
    };
    let next = if state.valid { state.scale.max(incoming) } else { incoming };
    if state.valid && next > state.scale {
        flags.align = true;
        flags.align_flush |= next - state.scale >= 64;
        for value in &mut state.values {
            *value = round_shift(*value, (next - state.scale) as u32, flags);
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
            flags.term_round = true;
            flags.term_flush |= -shift >= 64;
            round_shift(products[j], (-shift) as u32, flags)
        };
        state.values[j] = state.values[j].checked_add(term).expect("INT64 overflow");
    }
    let valid = state.values.iter().any(|x| *x != 0);
    flags.restart |= !valid;
    state.valid = valid;
    state.scale = if valid { next } else { 0 };
}

pub fn encode_state(state: &State) -> [u8; 272] {
    let mut output = [0u8; 272];
    output[0..2].copy_from_slice(&(state.processed as u16).to_le_bytes());
    output[2..4].copy_from_slice(&state.scale.to_le_bytes());
    output[4] = u8::from(state.valid);
    for (j, value) in state.values.iter().enumerate() {
        output[16 + j * 8..24 + j * 8].copy_from_slice(&value.to_le_bytes());
    }
    output
}

pub fn finalize(state: &State, flags: &mut Flags) -> [u8; 48] {
    let maximum = state.values.iter().map(|x| x.unsigned_abs()).max().unwrap();
    flags.zero_out |= maximum == 0;
    let exponent = if maximum == 0 {
        0
    } else {
        let q = 63 - maximum.leading_zeros() as i32 + i32::from(state.scale) - 38;
        flags.q_high |= q > 127;
        flags.q_low |= q < -127;
        q.clamp(-127, 127)
    };
    let shift = exponent - i32::from(state.scale) + 32;
    let mut output = [0u8; 48];
    for j in 0..32 {
        let v = state.values[j];
        let code = if shift >= 0 {
            let code = round_shift(v, shift as u32, flags);
            flags.round_clamp |= code.abs() > 127;
            code
        } else if shift < -6 || v.unsigned_abs() > (127u64 >> -shift) {
            flags.code_sat |= v != 0;
            v.signum() * 127
        } else {
            flags.left_shift |= v != 0;
            v << -shift
        };
        output[j] = code.clamp(-127, 127) as i8 as u8;
    }
    output[32] = (exponent + 127) as u8;
    output
}

// Reads the operand windows of the bank images and writes the destination
// rows into `c_new`, exactly as the instruction defines them.
pub fn execute(
    cmd: &Command,
    a: &[u8],
    b: &[Vec<u8>],
    c_old: &[u8],
    c_new: &mut [u8],
    flags: &mut Flags,
) -> State {
    let blocks = cmd.k / 32;
    let stride = stride(cmd.k);
    assert!(cmd.k > 0 && cmd.k % 32 == 0);
    assert!(
        cmd.a_codes + cmd.k / 16 <= 1024
            && cmd.a_exp + blocks.div_ceil(16) <= 1024
            && cmd.b_base + 8 * stride <= 1024
            && cmd.c_base + 17 <= 1024
    );
    let mut state = State { values: [0; 32], scale: 0, valid: false, processed: 0 };
    if cmd.accumulate {
        let old = &c_old[cmd.c_base * 16..cmd.c_base * 16 + 272];
        state.processed = u16::from_le_bytes([old[0], old[1]]) as usize;
        state.scale = i16::from_le_bytes([old[2], old[3]]);
        state.valid = old[4] != 0;
        for j in 0..32 {
            state.values[j] = i64::from_le_bytes(old[16 + j * 8..24 + j * 8].try_into().unwrap());
        }
    }
    assert!(state.processed + blocks <= 1024);
    let codes: Vec<i8> = a[cmd.a_codes * 16..cmd.a_codes * 16 + cmd.k].iter().map(|x| *x as i8).collect();
    let rows: Vec<&[u8]> = (0..32)
        .map(|j| {
            let start = (cmd.b_base + j % 8 * stride) * 16;
            &b[j / 8][start..start + stride * 16]
        })
        .collect();
    let weights: Vec<Vec<i8>> = rows.iter().map(|row| row[..cmd.k].iter().map(|x| *x as i8).collect()).collect();
    for n in 0..blocks {
        let lanes: Vec<&[i8]> = weights.iter().map(|w| &w[n * 32..n * 32 + 32]).collect();
        let exps: Vec<u8> = rows.iter().map(|row| row[cmd.k + n]).collect();
        block(&mut state, &codes[n * 32..n * 32 + 32], a[cmd.a_exp * 16 + n], &lanes, &exps, flags);
    }
    state.processed += blocks;
    let at = cmd.c_base * 16;
    if cmd.finalize {
        c_new[at..at + 48].copy_from_slice(&finalize(&state, flags));
    } else {
        c_new[at..at + 272].copy_from_slice(&encode_state(&state));
    }
    state
}
