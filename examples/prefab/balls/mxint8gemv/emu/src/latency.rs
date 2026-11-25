// The cycles of one MXINT8GEMV instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle and ready again the cycle after. Delays outside the Ball belong to
// the bank and bus model, not here. From Mxint8gemvBall.scala, cycle 0 being
// the command handshake, B = K / 32 blocks:
//
// Prologue (rs2 bit 56, accumulate): idle -> oldReq in cycle 1; 17 old C rows
//   of oldReq + oldResp, the prefetch held (`loading` is false in
//   oldReq/oldResp): load is entered in cycle 35 instead of 1, 34 more.
//
// Blocks: block b is prefetched on all five read ports at once, one request
//   per port per cycle (resp.ready stays high while loading, so the bank queue
//   holds at most one row). Each B port reads N_b rows: 16 code rows, plus 8
//   exponent rows when b % 16 == 0 (expRefresh), so N_b = 24 or 16; the A port
//   reads 3 or 2 and finishes earlier. Block 0: requests in the first N_0 load
//   cycles, `received` full N_0 + 2 cycles after load is entered, when load
//   leaves to mac.
//   Leaving load in cycle X starts the prefetch of block b + 1 in X + 1, full
//   at X + N_{b+1} + 2, while block b computes for C_b cycles and is back in
//   load at X + C_b + 1:
//     C_b = mac 8 + selectScale + align + add + normalize + advance = 13
//     C_b = mac 8 + selectScale + advance = 10 when all 32 products of block b
//           are zero (selectScale sees no incoming scale)
//   so block b + 1 leaves load max(N_{b+1} + 2, C_b + 1) cycles after block b;
//   with N >= 16 the prefetch always bounds it.
//
// Tail, after the last block leaves load: its C cycles, then
//   17 rows (header + 16 value pairs) of storeReq + storeResp, 34 cycles, or
//   with rs2 bit 57 (finalize) encode 32 lanes and 3 rows, 32 + 6 cycles;
//   complete is valid the next cycle.
// The UVM tests of the RTL check every instruction against it.

/// One MXINT8GEMV instruction of K = `k` with `rs2` as the Ball receives it.
/// `a` is the A bank and `b` the four B group banks, 16-byte rows from row 0:
/// their codes decide which blocks have only zero products.
pub fn cycles(k: u64, rs2: u64, a: &[u8], b: [&[u8]; 4]) -> u64 {
    let k = k as usize;
    let blocks = k / 32;
    let a_base = (rs2 & 0x3fff) as usize * 16;
    let b_base = ((rs2 >> 28) & 0x3fff) as usize * 16;
    let stride = (k / 16 + blocks.div_ceil(16)) * 16;
    let compute = |block: usize| {
        let zero = (0..32).all(|j| {
            let column = b_base + j % 8 * stride + block * 32;
            (0..32)
                .map(|i| i32::from(a[a_base + block * 32 + i] as i8) * i32::from(b[j / 8][column + i] as i8))
                .sum::<i32>()
                == 0
        });
        if zero {
            10
        } else {
            13
        }
    };
    let rows = |block: usize| if block % 16 == 0 { 24 } else { 16 };
    let mut total = if rs2 & (1 << 56) != 0 { 34 } else { 0 };
    total += rows(0) + 2;
    for block in 1..blocks {
        total += (rows(block) + 2).max(compute(block - 1) + 1);
    }
    total += compute(blocks - 1) + if rs2 & (1 << 57) != 0 { 32 + 6 } else { 34 } + 1;
    total as u64
}
