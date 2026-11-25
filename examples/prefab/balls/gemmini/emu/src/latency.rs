// The cycles of one Gemmini instruction from its command handshake to the
// first cycle its completion is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle (the port takes its next write the cycle after), metadata reads
// answered the next cycle, every metadata write, SubROB row, completion and
// response accepted at once, and an idle mesh. Delays outside the Ball belong
// to the bank and bus model, not here. Cycle 0 is the command handshake.
// The UVM tests of the RTL check every instruction against it.
//
// Mesh (MeshWithDelays, Mesh; 16x16 PEs, tileLatency 0, outputDelay 1): a row
// whose a/b/d handshake is at cycle f enters the array at f + 1 (a_buf, b_buf,
// d_buf); PE (r, c) sees it at f + 1 + c + r + 1 (skew c, one register per
// tile row); the bottom row's output takes the outputDelay register and the
// de-skew 15 - c: io.resp is valid 32 cycles after the row enters. After a
// flush request a flush row enters every cycle, without operands.
//
// BLink commands (GemminiExCtrl) select by rs2[3:0]; `ws`, `a_transpose`,
// `b_transpose` are the configuration the last GEMMINI_CONFIG set (reset: OS,
// no transpose). n = iter, 0 meaning 16; total_rows holds at most 16.
//   CONFIG (0): sIdle -> sCommit, the response at 1.
//   FLUSH (4): sIdle -> sFlush, the idle mesh takes the flush request and the
//     response fires with it: 1.
//   PRELOAD (1): sPreloadRead requests rows at 1..n and sees the count at
//     n + 1; sPreloadFeed sends the mesh request at n + 2. Feeding from the
//     read queue (its SRAM prefetch gives a row a cycle) at n + 3..2n + 2, the
//     count at 2n + 3 answers: 2n + 3. WS with b_transpose first captures the
//     n rows (n + 3..2n + 2), sets xpose_ready at 2n + 3, feeds their columns
//     at 2n + 4..3n + 3: 3n + 4.
//   COMPUTE_PRELOADED (2): sComputeRead reads a row of op1 and op2 (or none,
//     zeroed) each cycle at 1..n, the count at n + 1; sComputeFeed sends the
//     mesh request at F = n + 2.
//   COMPUTE_ACCUMULATED (3): always OS. sPreloadFeed sends the zero preload at
//     1 and feeds it without reads at 2..n + 1, the count at n + 2; then the
//     reads at n + 3..2n + 2, the count at 2n + 3, the request at F = 2n + 4.
//   Both then: OS without a_transpose buffers op1 (op1FromBuf), OS with
//     b_transpose buffers op2 (zero_op2, rs2[4], zeroes only the bias D of a
//     WS compute; in OS op2 is B and always read): rows at F + 1..F + n (a
//     zero_op1_tail row of op1 is buffered as zero, without a read),
//     xpose_ready at F + n + 1, rows fed at f0 = F + n + 2 onwards; otherwise
//     fed from the queues at f0 = F + 1 onwards; n rows, the count at
//     d = f0 + n.
//     OS: sComputeFlush sends the flush at d + 1, its rows enter at
//     d + 2..d + 17 and come out at d + 34..d + 49; they are the only rows
//     with a tag and total_rows 16 (no total_rows entry), so n must be 16;
//     d + 50 sees them all, sStore from s = d + 51.
//     WS: sDrain takes the compute's own rows (tagged by the PRELOAD before
//     it; that PRELOAD's rows, still leaving the mesh, carry a garbage tag),
//     out at f0 + 33..f0 + n + 32, sees n at f0 + n + 33: sStore from
//     s = d + 34.
//     sStore writes the four groups of a row together every 2 cycles, the
//     last at s + 2n - 2, counts it at s + 2n - 1, sees n at s + 2n,
//     sCommit at s + 2n + 1. COMPUTE_ACCUMULATED reads the row's four groups
//     of the old destination first (sReadAccum: request, answer, dequeue,
//     3 cycles each): 14 cycles a row, sCommit at s + 14n + 1.

/// One BLink command of GemminiBall: its iter and rs2, and the configuration.
pub fn cycles(iter: u64, rs2: u64, ws: bool, a_transpose: bool, b_transpose: bool) -> u64 {
    let n = if iter == 0 { 16 } else { iter };
    assert!(n <= 16, "gemmini: iter {iter} exceeds the 16 rows of the array");
    match rs2 & 0xf {
        0 | 4 => 1,
        1 => {
            if ws && b_transpose {
                3 * n + 4
            } else {
                2 * n + 3
            }
        }
        sub @ (2 | 3) => {
            let accumulated = sub == 3;
            let os = accumulated || !ws;
            let request = if accumulated { 2 * n + 4 } else { n + 2 };
            let op1_from_buf = os && !a_transpose;
            let op2_from_buf = os && b_transpose;
            let feed = if op1_from_buf || op2_from_buf { request + n + 2 } else { request + 1 };
            let fed = feed + n;
            let store = if os {
                assert!(n == 16, "gemmini: an OS compute collects only 16-row flushes");
                fed + 51
            } else {
                fed + 34
            };
            if accumulated {
                store + 14 * n + 1
            } else {
                store + 2 * n + 1
            }
        }
        sub => panic!("gemmini: rs2 sub-command {sub} is no instruction"),
    }
}

// SubROB expansion commands (Loop, LoopMatmulUnroller, LoopConvUnroller):
//   a loop configuration: storing writes its 8 metadata bytes at 1..8,
//     configured completes at 9.
//   a loop: reading takes a metadata byte in 2 cycles (request 1 + 2b, answer
//     2 + 2b); starting hands the bounds to the unroller the cycle after the
//     last answer; streaming passes a row each cycle from the next one.

/// A loop configuration command.
pub fn loop_config_cycles() -> u64 {
    9
}

/// GEMMINI_LOOP_WS: 56 metadata bytes, the last answer at 112, start at 113,
/// rows from 114: alloc, prime load, per tile [preload + mvin B, compute,
/// mvout + mvin A] without the third row after the last tile, drain, free:
/// 3N + 3 rows, the last at 3N + 116.
pub fn loop_ws_cycles(max_i: u64, max_j: u64, max_k: u64) -> u64 {
    3 * (max_i * max_j * max_k) + 116
}

/// GEMMINI_LOOP_CONV_WS: 72 metadata bytes, the last answer at 144, start at
/// 145, rows from 146: alloc, per output tile (batch, row, column, 16 output
/// channels) K compute tiles (kernel row, column, 16 input channels) of
/// [mvin weight, preload + mvin input, compute] and an mvout, free:
/// 2 + T(3K + 1) rows, the last at 147 + T(3K + 1).
pub fn loop_conv_ws_cycles(batch: u64, out_dim: u64, out_channels: u64, in_channels: u64, kernel_dim: u64) -> u64 {
    let tiles = batch * out_dim * out_dim * out_channels.div_ceil(16);
    let per_tile = kernel_dim * kernel_dim * in_channels.div_ceil(16);
    147 + tiles * (3 * per_tile + 1)
}

// The Ball instructions a loop expands into run as separate commands of this
// Ball, each with its own cycles above; DIM = 16 rows each.

/// The PRELOADs and COMPUTEs of GEMMINI_LOOP_WS (LoopMatmulUnroller): per
/// (i, j) a PRELOAD and a COMPUTE_PRELOADED at k = 0, a COMPUTE_ACCUMULATED
/// for every later k.
pub fn loop_ws_ball_cycles(
    max_i: u64,
    max_j: u64,
    max_k: u64,
    ws: bool,
    a_transpose: bool,
    b_transpose: bool,
) -> u64 {
    let first = cycles(16, 1, ws, a_transpose, b_transpose) + cycles(16, 2, ws, a_transpose, b_transpose);
    let later = if max_k > 1 {
        (max_k - 1) * cycles(16, 3, ws, a_transpose, b_transpose)
    } else {
        0
    };
    max_i * max_j * (first + later)
}

/// The PRELOADs and COMPUTEs of GEMMINI_LOOP_CONV_WS (LoopConvUnroller): per
/// compute tile a PRELOAD and a COMPUTE with zero_op1_tail: PRELOADED (with
/// zero_op2 = no_bias) for the first tile of an output tile, ACCUMULATED after.
#[allow(clippy::too_many_arguments)]
pub fn loop_conv_ws_ball_cycles(
    batch: u64,
    out_dim: u64,
    out_channels: u64,
    in_channels: u64,
    kernel_dim: u64,
    no_bias: bool,
    ws: bool,
    a_transpose: bool,
    b_transpose: bool,
) -> u64 {
    let tiles = batch * out_dim * out_dim * out_channels.div_ceil(16);
    let per_tile = kernel_dim * kernel_dim * in_channels.div_ceil(16);
    let preload = cycles(16, 1, ws, a_transpose, b_transpose);
    let first = cycles(16, 2 | (no_bias as u64) << 4 | 1 << 5, ws, a_transpose, b_transpose);
    let later = if per_tile > 1 {
        (per_tile - 1) * cycles(16, 3 | 1 << 5, ws, a_transpose, b_transpose)
    } else {
        0
    };
    tiles * (per_tile * preload + first + later)
}
