// The cycles of one MXMM instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle. Delays outside the Ball belong to the bank and bus model, not here.
// From Unit.scala, InputWindow.scala, Panels.scala, Array.scala and PE.scala
// (16x16 tile, arithmeticLatency 3, 7 accumulator contexts):
//   2        idle -> waitChannels -> loadInputs.
//   load     both operands at once, the slower one counts:
//            a linear operand of R rows: 2 per code word (request, response),
//              R * K / 16 words, then its scales from the next (aligned)
//              word: 2 per scale word plus one cycle per scale byte;
//            the window operand A (InputWindow): 2 per code word, then per row
//              a scale word request (2) at its first byte b and one more for
//              each word boundary its K / 32 bytes cross, so
//              ceil((b + K / 32) / 16) requests, plus one cycle per byte.
//   1        the cycle loadInputs sees both done -> initRequest.
//   per batch of c of the C panels (M / 16, or 1 for M = 1, panel rows times
//   N / 16; 7 at a time), period p = max(c, arithmeticLatency + 1):
//     init    16 rows per context, one cycle each; without first the
//             panelHeight rows (16, or 1 for M = 1) read the accumulator SRAM
//             and take a second cycle (initRequest, initResponse).
//     p (K - 1) + c - 1  compute up to the last panel read.
//     5       Panels RegNext (1) + PE three stages (3), completed seen, then
//             store the next cycle.
//     16 c    store, one row per cycle.
//   last     10 per C line, M * N / 16 lines: outputRequest, outputResponse,
//            4 x (outputWrite, outputAck); without last, complete follows
//            the store at once.
// The UVM tests of the RTL check it exactly on every instruction.

/// One MXMM instruction as the Ball receives it: C (m x n) += A (m x k) B,
/// `window` = Some((fullK, startK)) for MXMM_MXFP8_WINDOW.
pub fn cycles(m: u64, n: u64, k: u64, first: bool, last: bool, window: Option<(u64, u64)>) -> u64 {
    let blocks = k / 32;
    let b_load = 2 * n * k / 16 + 2 * (n * blocks).div_ceil(16) + n * blocks;
    let a_load = match window {
        None => 2 * m * k / 16 + 2 * (m * blocks).div_ceil(16) + m * blocks,
        Some((full_k, start_k)) => {
            let requests: u64 = (0..m)
                .map(|row| ((m * full_k + row * (full_k / 32) + start_k / 32) % 16 + blocks).div_ceil(16))
                .sum();
            2 * m * k / 16 + 2 * requests + m * blocks
        }
    };
    let panel_height = if m == 1 { 1 } else { 16 };
    let panels = if m == 1 { 1 } else { m / 16 } * (n / 16);
    let mut cycles = 3 + a_load.max(b_load);
    for batch in (0..panels).step_by(7) {
        let c = (panels - batch).min(7);
        let p = c.max(4);
        cycles += c * (16 + if first { 0 } else { panel_height });
        cycles += p * (k - 1) + c - 1 + 5 + 16 * c;
    }
    if last {
        cycles += 10 * m * (n / 16);
    }
    cycles
}
