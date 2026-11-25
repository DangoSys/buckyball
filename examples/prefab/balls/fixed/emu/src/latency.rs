// The cycles of one FIXED instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle and the port free again the cycle after. Delays outside the Ball belong
// to the bank and bus model, not here. From FixedBall.scala, cycle 0 being the
// command handshake:
//
// Shared terms
//   1       idle -> the first state.
//   coef    coefficientRequest/coefficientResponse on port 1, 2 cycles a row:
//           8 rows once (16) when not streaming, 2 rows before every row (4)
//           when streaming (rs2[9]).
//   read 3  reading: request, response, and since `received` is a register
//           the move out of reading is seen one cycle later.
//   mul 5   Multiply(32): request taken while idle, four radix-256 steps,
//           complete the fifth cycle when the response is taken.
//   write 2 writing, acknowledge; complete follows the last acknowledge.
//
// Modes (rs2[2:0]), per row after its coefficients:
//   0 requant   read 3, prepare 1 (four lanes into four multipliers),
//               scaleWait 5, write 2 = 11
//   1 hswish    read 3, four lanes, write 2; a lane is prepare 1, scaleWait 5
//               unless M = 1 and shift = 0 (prepare copies a), activate 1,
//               then unless q = 0 or |q| >= 3Q: a second multiply 5 and
//               divide 4, pack 1 (so 17 at most, 3 at least)
//   2 hsigmoid  as hswish without the second multiply: divide 4 (12 at most)
//   3 mul       read 3 (both ports together), four lanes of prepare 1,
//               productWait 5, pack 1 = 28, write 2 = 33
//   4 relu,
//   5 add       read 3, prepare 1 (four lanes at once), write 2 = 6
//   6 sum       wideRequest/wideResponse loads the 8 partial-sum rows (16);
//               per row read 3, sumAccumulate 1 = 4; then 8 rows written (16)
//   7 mean      12 wide rows (8 sums, 4 divisors) (24), 16 channels of
//               meanPrepare 1 + meanDivide 64, then `iter` (4) rows written
//
// q = round(a * M / 2^shift) (ties to even, saturated to I32) with M and shift
// of the lane's channel: B rows 0..3 hold the sixteen M and rows 4..7 the
// sixteen shifts, channel (row % 4) * 4 + lane; streaming, B row 2r holds the
// four M of row r and row 2r + 1 its four shifts, channel = lane.
// The UVM tests of the RTL check it on every instruction.

/// One FIXED instruction as the Ball receives it: `iter` rows, `rs2`, and the
/// words of its A and B banks (four I32 per row, row-major), which only
/// HSWISH and HSIGMOID read.
pub fn cycles(iter: u64, rs2: u64, a: &[i32], b: &[i32]) -> u64 {
    let mode = rs2 & 7;
    let fraction = (rs2 >> 4) & 31;
    let stream = rs2 & 512 != 0;
    match mode {
        6 => 1 + 16 + 4 * iter + 16,
        7 => 1 + 24 + 16 * (1 + 64) + 2 * iter,
        _ => {
            let coefficients = if mode <= 2 && !stream { 16 } else { 0 };
            let mut cycles = 1 + coefficients;
            for row in 0..iter as usize {
                let lanes: u64 = match mode {
                    0 => 1 + 5,
                    1 | 2 => (0..4).map(|lane| lane_cycles(mode, fraction, stream, row, lane, a, b)).sum(),
                    3 => 4 * (1 + 5 + 1),
                    _ => 1,
                };
                cycles += if stream { 4 } else { 0 } + 3 + lanes + 2;
            }
            cycles
        }
    }
}

/// One HSWISH or HSIGMOID lane: prepare, scale, activate, its function, pack.
fn lane_cycles(mode: u64, fraction: u64, stream: bool, row: usize, lane: usize, a: &[i32], b: &[i32]) -> u64 {
    let (m, shift) = if stream {
        (b[row * 8 + lane], b[row * 8 + 4 + lane])
    } else {
        let channel = (row % 4) * 4 + lane;
        (b[channel], b[16 + channel])
    };
    // The RTL keeps M[30:0] and shift[5:0].
    let m = m & 0x7fff_ffff;
    let shift = (shift & 63) as u32;
    let product = i64::from(a[row * 4 + lane]) * i64::from(m);
    let magnitude = product.unsigned_abs();
    let divisor = 1u64 << shift;
    let quotient = magnitude >> shift;
    let remainder = magnitude & (divisor - 1);
    let rounded = quotient + u64::from(remainder * 2 > divisor || (remainder * 2 == divisor && quotient & 1 != 0));
    let q = if product < 0 { -(rounded as i128) } else { rounded as i128 }.clamp(i32::MIN as i128, i32::MAX as i128);
    let threshold = 3i128 << fraction;
    let skip_scale = m == 1 && shift == 0;
    let saturated = q == 0 || q <= -threshold || q >= threshold;
    let function = if saturated {
        0
    } else if mode == 1 {
        5 + 4
    } else {
        4
    };
    1 + if skip_scale { 0 } else { 5 } + 1 + function + 1
}
