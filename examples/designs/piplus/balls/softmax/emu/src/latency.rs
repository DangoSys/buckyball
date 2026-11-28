// The cycles of one SOFTMAX instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle (the port free the cycle after), the response accepted at once.
// From SoftmaxBall.scala and Exp.scala, L = ceil(W / 4) bank rows a row, a row
// from the first cycle S of its pass 0:
//   L + 2   pass 0: read S .. S+L-1, answered S+1 .. S+L, received === L and
//           passDone at S+L+1, pass 1 from S+L+2
//   L + 19  pass 1: answered from its cycle 1 on; Exp registers d, u, n, f,
//           two per Horner step (6) and e: 17 stages, so produced === L
//           17 + L + 1 cycles in, passDone, pass 2 the cycle after
//   Nd + 1  pass 2 starts 1 / s at its first cycle, valid Nd later,
//           inverseOk the next: the first write (Exp filled the queue at its
//           cycle 18, then holds)
//   2L      a write every 2 cycles (accept, acknowledge), the queue and the
//           stalled Exp pipeline always holding the next row; written === L
//           and passDone 2 cycles after the last write
//   1       the next row's pass 0 starts, or done raises the response
// so a row is 4L + Nd + 23, the response 1 + the rows.
// DivSqrtRecFN_small (hardfloat) divides in sigWidth + 2 = 26 cycles from the
// cycle it enters to its out valid, 25 when the dividend's significand is not
// below the divisor's (skipCycle2: 1 / s for s a power of two). s, the Ball's
// own FP32 sum of the exponentials, operation for operation as softmax.rs, is
// at least 1 (the maximum's e is 1) and finite: never a special operand.
// The UVM tests of the RTL check it exactly.

const LOG2E: f32 = 1.442_695_040_888_963_4;
/// P(f) ~ 2^f on [0, 1): c1 .. c6.
const COEFFICIENTS: [f32; 6] = [
    0.693_147_180_559_945_3,
    0.240_226_506_959_100_7,
    0.055_504_108_664_821_6,
    0.009_618_129_107_628_5,
    0.001_333_355_814_642_8,
    0.000_154_035_303_933_8,
];

/// 2^u of u <= 0, zero below -126.
fn exp2(u: f32) -> f32 {
    if u < -126.0 {
        return 0.0;
    }
    let n = u.floor();
    let f = u - n;
    let mut p = COEFFICIENTS[5];
    for c in COEFFICIENTS[..5].iter().rev() {
        p = p * f + c;
    }
    p = p * f + 1.0;
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// The four independent FP32 lanes and their reduction order mirror the RTL.
pub(super) fn exponentials(width: usize, scaled: impl Fn(usize) -> f32, mut emit: impl FnMut(usize, f32)) -> f32 {
    let maximum = (0..width).map(&scaled).fold(f32::NEG_INFINITY, f32::max);
    let mut lanes = [0.0f32; 4];
    for i in 0..width {
        let e = exp2((scaled(i) - maximum) * LOG2E);
        emit(i, e);
        lanes[i % 4] += e;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// One SOFTMAX instruction as the Ball receives it (`iter` rows, rs2) on the
/// words of its input bank.
pub fn cycles(iter: u64, rs2: u64, input: &[u32]) -> u64 {
    cycles_with(iter, rs2, |index| f32::from_bits(input[index]))
}

pub(super) fn cycles_with(iter: u64, rs2: u64, read: impl Fn(usize) -> f32) -> u64 {
    let w = (rs2 & 0xffff) as usize;
    let in_base = ((rs2 >> 16) & 0x3fff) as usize;
    let scale = f32::from_bits(((((rs2 >> 44) & 0xff) as u8 as i8 as i32 + 127) as u32) << 23);
    let lines = w.div_ceil(4);
    let mut total = 1;
    for row in 0..iter as usize {
        let at = (in_base + row * lines) * 4;
        let s = exponentials(w, |i| read(at + i) * scale, |_, _| {});
        let nd = if s.to_bits() & 0x7f_ffff == 0 { 25 } else { 26 };
        total += 4 * lines as u64 + nd + 23;
    }
    total
}
