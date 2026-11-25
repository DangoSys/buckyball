// The cycles of one MXINT8MM instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle and ready again the cycle after. Delays outside the Ball belong to
// the bank and bus model, not here. From Mxint8matmulBall.scala, cycle 0 being
// the command handshake, on a `rows` x `cols` PE array with `lanes` code pairs
// per PE per cycle (32 / lanes mac steps), r active rows and c active columns
// of a tile:
//   1        idle -> tile: the first tile in cycle 1
//   per tile: tile 1, [oldC], then per K block b, then store
//     oldC   R + 2, only with rs2 bit 62: R = r * c / 4 old C rows on port A,
//            one request per cycle from the first oldC cycle, the last
//            response R cycles after the first request, the counter seen full
//            one cycle later, and oldC leaves that cycle
//     load   N_b + 2: ports A and B issue one request per cycle from the first
//            load cycle (resp.ready high in load), N_b = max(3 r, 3 c) when
//            b % 16 == 0 (two code rows + exponent row each), else
//            max(2 r, 2 c); the last response lands N_b cycles after the
//            first request, the response counters are seen full one cycle
//            later, and load leaves that cycle
//     mac 32 / lanes + convert 1 + accumulate 1
//     store  2 W + 1: W = r * c / 4 writes, one every 2 cycles (acknowledged
//            the next cycle, the port ready again the cycle after), the
//            counters seen full one cycle after the last acknowledgement
//   the last store leaves to complete, whose response is valid that cycle.
// The PE has no data-dependent timing.
// The UVM tests of the RTL check every instruction against it.

/// One MXINT8MM instruction: C[m, n] over K = `k`, `from_old` = rs2 bit 62.
pub fn cycles(m: u64, n: u64, k: u64, from_old: bool, rows: u64, cols: u64, lanes: u64) -> u64 {
    let mut total = 1;
    for m0 in (0..m).step_by(rows as usize) {
        for n0 in (0..n).step_by(cols as usize) {
            let r = rows.min(m - m0);
            let c = cols.min(n - n0);
            let w = r * c / 4;
            total += 1 + 2 * w + 1;
            if from_old {
                total += w + 2;
            }
            for block in 0..k / 32 {
                let parts = if block % 16 == 0 { 3 } else { 2 };
                total += parts * r.max(c) + 2 + 32 / lanes + 2;
            }
        }
    }
    total
}
