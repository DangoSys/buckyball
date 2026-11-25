// The cycles of one LAYERNORM instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle (the port free the cycle after), the response accepted at once.
// From LayerNormBall.scala, L = ceil(W / 4) bank rows a row; a LayerNorm row
// from the first cycle S of its pass 0:
//   L + 2  pass 0: x read S .. S+L-1, answered S+1 .. S+L, received === L
//          and passDone at S+L+1, pass 1 from S+L+2
//   L + 2  pass 1: the same
//   Ns     pass 2 starts sqrt(var + eps) at its first cycle, valid Ns later
//   1 + Nd rootOk the next cycle starts 1 / root, valid Nd later
//   1      rOk: the first output enqueues (x, gamma and beta are there)
//   3L     one bank row every 3 cycles: the enqueue clears gammaOk/betaOk,
//          gamma is taken the next cycle and beta the one after on read port
//          1, the enqueue the third; the last one is written a cycle after,
//          acked the next, written === L and passDone 3 cycles after it
//   1      the next row's pass 0 starts, or done raises the response
// so a row is 5L + Ns + Nd + 7. Pass 0 of the first row also waits for 1 / W,
// started at cycle 1 and valid Nw later (inverseOk at 2 + Nw against L + 2):
// max(0, Nw - L) more.
// RMSNorm (rs2 bit 58): pass 0 reads nothing and ends at its first cycle once
// inverseOk and epsOk: 1 cycle, the first row's at cycle 2 + Nw (Nw + 1
// more); pass 2 reads gamma only, two rows ahead, so an output every 2 cycles
// (gamma taken the cycle after the enqueue, the enqueue the next), the write
// port's own rate; written === L 3 cycles after the last enqueue: a row is
// 1 + (L + 2) + Ns + Nd + 2 + (2L + 1) + 1 = 3L + Ns + Nd + 7.
// DivSqrtRecFN_small (hardfloat), from the cycle it enters to its out valid:
// a division sigWidth + 2 = 26 cycles, 25 when the dividend's significand is
// not below the divisor's (skipCycle2: 1 / x for x a power of two); a square
// root sigWidth + 1 = 25 for an even exponent, 24 for an odd one; 1 for a
// special operand (zero, infinite, NaN, a negative square root). var + eps is
// the Ball's own FP32 sum, operation for operation as layernorm.rs.
// The UVM tests of the RTL check it exactly.

/// Lane sums in increasing element order, then (s0 + s1) + (s2 + s3).
fn sum(values: impl Iterator<Item = f32>) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (i, v) in values.enumerate() {
        lanes[i % 4] += v;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// The exponent of a finite nonzero value, subnormals normalized.
fn exponent(v: f32) -> i32 {
    let bits = v.to_bits();
    let field = ((bits >> 23) & 0xff) as i32;
    if field == 0 {
        31 - (bits & 0x7f_ffff).leading_zeros() as i32 - 149
    } else {
        field - 127
    }
}

fn special(v: f32) -> bool {
    v == 0.0 || !v.is_finite()
}

/// 1 / b of a finite nonzero b: a power of two skips a cycle.
fn reciprocal(b: f32) -> u64 {
    let bits = b.to_bits() & 0x7fff_ffff;
    let fraction = bits & 0x7f_ffff;
    let power = if bits >> 23 == 0 {
        fraction.is_power_of_two()
    } else {
        fraction == 0
    };
    if power {
        25
    } else {
        26
    }
}

/// One LAYERNORM instruction as the Ball receives it (`iter` rows, rs2) on the
/// words of its input bank and the eps word of its parameter bank.
pub fn cycles(iter: u64, rs2: u64, input: &[u32], eps: u32) -> u64 {
    let w = (rs2 & 0xffff) as usize;
    let in_base = ((rs2 >> 16) & 0x3fff) as usize;
    let rms = (rs2 >> 58) & 1 == 1;
    let lines = w.div_ceil(4);
    let l = lines as u64;
    let inverse = 1.0 / w as f32;
    let eps = f32::from_bits(eps);
    let nw: u64 = if w.is_power_of_two() { 25 } else { 26 };
    let mut total = 1 + if rms { nw + 1 } else { nw.saturating_sub(l) };
    for row in 0..iter as usize {
        let at = (in_base + row * lines) * 4;
        let x: Vec<f32> = input[at..at + w].iter().map(|&b| f32::from_bits(b)).collect();
        let mean = if rms { 0.0 } else { sum(x.iter().copied()) * inverse };
        let shifted = sum(x.iter().map(|&v| (v - mean) * (v - mean))) * inverse + eps;
        let (ns, nd) = if special(shifted) || shifted < 0.0 {
            (1, 1)
        } else {
            (
                if exponent(shifted).rem_euclid(2) == 0 { 25 } else { 24 },
                reciprocal(shifted.sqrt()),
            )
        };
        total += if rms { 3 * l } else { 5 * l } + ns + nd + 7;
    }
    total
}
