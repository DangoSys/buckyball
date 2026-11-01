use super::accumulator::Format;
use bebop_bemu::thread::instructions::decode::rs1_iter;

pub(crate) fn latency(xs1: u64, xs2: u64, format: Format) -> u64 {
    let m = xs2 & 0xfff;
    let n = (xs2 >> 12) & 0xfff;
    let k = rs1_iter(xs1);
    let mx = matches!(format, Format::Mxfp8 | Format::Mxfp8Window);
    let panel_height = if m == 1 { 1 } else { 16 };
    let panels = m.div_ceil(16) * (n / 16);
    let mut a_load = 5 * m * k / if mx { 16 } else { 4 };
    let mut b_load = 5 * n * k / if mx { 16 } else { 4 };
    if mx {
        let blocks = k / 32;
        b_load += n * blocks + 5 * (n * blocks).div_ceil(16);
        if format == Format::Mxfp8Window {
            let full_k = (xs2 >> 32) & 0xffff;
            let start_k = xs2 >> 48;
            let reads: u64 = (0..m)
                .map(|row| {
                    let byte = (m * full_k + row * (full_k / 32) + start_k / 32) & 15;
                    (byte + blocks).div_ceil(16)
                })
                .sum();
            a_load += m * blocks + 5 * reads;
        } else {
            a_load += m * blocks + 5 * (m * blocks).div_ceil(16);
        }
    }
    let first = xs2 & (1 << 24) != 0;
    let last = xs2 & (1 << 25) != 0;
    let delay = if format == Format::F32 { 7 } else { 4 };
    let mut cycles = 10 + a_load.max(b_load);
    // Seven independent accumulator contexts hide the three-stage FMA.
    for batch in (0..panels).step_by(7) {
        let count = (panels - batch).min(7);
        let period = count.max(delay);
        cycles += count * (32 + if first { 0 } else { panel_height });
        cycles += k * period + count + delay - period;
    }
    if last {
        cycles += 18 * m * (n / 16);
    }
    cycles
}
