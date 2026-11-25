// The cycles of one SMatMul instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle and its port free again the cycle after, the response accepted at
// once. Delays outside the Ball belong to the bank and bus model, not here.
// From SMatMulUnit.scala and Array.scala; the command is accepted on cycle 0.
// The UVM tests of the RTL check every instruction against it exactly.

/// SMATMUL_BIAS (iter is always four rows):
///   2       idle -> waitForChannels -> biasReadReq: the first read request
///   2 / row biasReadReq (request), biasReadResp (answer, the next cycle)
///   complete follows the fourth answer: 2 + 2 * 4.
pub fn bias_cycles() -> u64 {
    2 + 2 * 4
}

/// SMATMUL_OS as the Ball receives it: K = `iter` per reduction group,
/// rs2 = { C base [31:26], last [25], first [24], N [23:12], M [11:0] },
/// `k_groups` = op1_col reduction groups, tile_rows x tile_cols array.
/// a = rows of one output tile (1 for M = 1, else tile_rows), T = output tiles
/// (1 for M = 1, else M / tile_rows), R = K / tile_cols reduction tiles:
///   2            idle -> waitForChannels -> initAccumulator or loadTile
///   first: a * T initAccumulator, one accumulator row per cycle
///   T * R * k_groups tile iterations, each:
///     max(a, tile_cols) + 2  loadTile: A (a rows) and B (tile_cols rows)
///                  requested one per cycle, answered one cycle later, both
///                  counts full the cycle after the last answer, which starts
///                  the array and leaves for runArray
///     tile_rows + tile_cols + 16  runArray: the array counts 0..lastCyc,
///                  lastCyc = tile_rows + tile_cols + 15, done on lastCyc
///     2 * a        readAccumulator, writeAccumulator per accumulator row
///   last: 10 * M write back, per C row readResult, holdResult, then four
///                  words of writeResult (request) and waitForCWrite (answer)
///   then complete: the response is valid. The C groups (wr_col) only route
///   the writes; they add no cycle.
pub fn os_cycles(iter: u64, rs2: u64, k_groups: u64, tile_rows: u64, tile_cols: u64) -> u64 {
    let m = rs2 & 0xfff;
    let first = (rs2 >> 24) & 1;
    let last = (rs2 >> 25) & 1;
    let a = if m == 1 { 1 } else { tile_rows };
    let tiles = if m == 1 { 1 } else { m / tile_rows };
    let reductions = iter / tile_cols;
    let tile = a.max(tile_cols) + 2 + tile_rows + tile_cols + 16 + 2 * a;
    2 + first * a * tiles + tiles * reductions * k_groups * tile + last * 10 * m
}
