// The cycles of one REDMULE_F16 instruction from its command handshake
// (cycle 0) to the first cycle its response is valid, against an ideal outside
// world: every bank request ready, a read answered the next cycle, a write
// acknowledged the next cycle, the response accepted at once. Delays outside
// the Ball belong to the bank and bus model, not here. From RedmuleBall.scala
// and the native RedMulE it wraps (Height 8, Width 8, NumPipeRegs 1, one
// 256-bit HCI port: 16 FP16 per access, two bank rows).
//
// RedmuleBall runs one native job per row tile of 8 C rows, each starting
// with a soft clear, so every tile starts from the same native state. A tile:
// Start, 28 cycles before the first native request:
//   1      channels (the first tile; later tiles enter configure directly)
//   20     10 control-port steps, configure + controlReply each (the HWPE
//          target grants at once, r_valid the next cycle): SOFT_CLEAR,
//          ACQUIRE, 7 job registers, COMMIT_TRIGGER; running at 22
//   +4     commit_trigger write at 20 -> job_trigger at 21 -> config FIFO ->
//          tiler valid_o at 25 (its input, x_rows*w_cols and *_iter
//          registers) -> memory scheduler config FIFOs non-empty at 26:
//          req_start of the X, W (and Y) sources
//   +3     addressgen addr_valid_q at 27, source address FIFO 28,
//          hci_core_fifo outgoing 29: the wrapper latches it in busIdle at 29
// One native load, two bank rows (bankRow, bankRow + 1), the bus FSM:
//   busIdle, (readRequest, readReply, advance) for a row inside the window or
//   (readRequest, advance) for a row outside it, grant, response:
//   9 both rows inside, 8 one, 7 none.
// One native store (z_strb covers 8 columns, so the second row's mask is 0):
//   busIdle, writeRequest, writeReply, advance (row inside the C window) or
//   writeRequest, advance (outside), writeRequest, advance (mask 0), grant:
//   7 row inside, 6 outside.
// Work of a tile of m <= 8 rows (A rows tile_a.., C rows tile_c..):
//   X: m * ceil(K / 16) loads, A row r chunk c at tile_a + r * K / 8 + 2c,
//      window [tile_a, tile_a + m * K / 8)
//   W: n = K loads, or 16 when K <= Height (the tiler promotes N <= Height to
//      MinimumSizeN = 16 W rows), at b_base + i, window [b_base, b_base + K)
//   Y (first = 0, GEMM: the previous C): 8 loads at tile_c + j, window
//      [tile_c, tile_c + m)
//   Z: 8 stores (Width rows) at tile_c + j, window [tile_c, tile_c + m)
// Once the engine runs it takes a W row every LOAD_W + WAIT (2 cycles),
// faster than any access, so a stream always holds a queued request: every
// access runs back to back.
// GEMM bubble: the scheduler stays in PRELOAD until the z buffer holds the 8
//   Y rows; till then nothing drains the W stream (credit 8: its
//   hci_core_fifo incoming and w_buffer FIFOs) nor the X stream beyond its
//   first chunk (credit m + 8). The memory mux grants, from the soft-cleared
//   counter rr = 0, the first requesting stream among rr, rr + 1, ... (X 0,
//   W 1, Y 2, Z 3), rr + 1 after every grant. If neither X nor W can request
//   after the last Y grant, the bus idles: the last Y row reaches the z buffer
//   at R+2 (R its response), LOAD_W at R+3 pops the W FIFO, the space reaches
//   the hci_core_fifo at R+4 and the next request at R+5: 4 idle cycles.
// Tail, 14 cycles from the cycle after the last load's response (R) to the
// first store request:
//   R+2    the W row through the hci_core_fifo incoming and w_buffer FIFOs,
//          consumed in LOAD_W: w_cols_iter_en, w_done
//   R+3..  z_wait_counter 0, 1 (NumPipeRegs) -> z_avail_en at R+5
//   R+5..  z_avail / fill for z_height = 8 cycles, z_avail_clr at R+12
//   R+13   z buffer PUSHED: z_valid
//   R+15   z_buffer FIFO, store hci_core_fifo: the wrapper latches it
// End, 4 cycles: last store grant G; store FIFO empty at G+1 -> ctrl
//   FINISHED at G+2 -> IDLE at G+3, busy_o low (evt already set `finished`):
//   the last tile completes at G+4, the response valid; another tile enters
//   configure at G+4, 2 cycles later in its own count than the first.
// Tile: 28 + every access + GEMM bubble + 14 + 4; the instruction: the sum
// over its tiles, minus 2 for every tile after the first.

/// One REDMULE_F16 instruction as the Ball receives it: M = rs2[11:0] rows,
/// K = iter, the A, B and C row bases rs2[21:12], rs2[31:22], rs2[41:32] and
/// first = rs2[42] (C not read).
pub fn cycles(rows: u64, k: u64, a_base: u64, b_base: u64, c_base: u64, first: bool) -> u64 {
    let n = if k <= 8 { 16 } else { k };
    let chunks = k.div_ceil(16);
    let row_cost = |row: u64, lo: u64, len: u64| if row >= lo && row < lo + len { 3 } else { 2 };
    let load = |row: u64, lo: u64, len: u64| 3 + row_cost(row, lo, len) + row_cost(row + 1, lo, len);
    let mut total = 0;
    for tile in 0..rows.div_ceil(8) {
        let m = (rows - 8 * tile).min(8);
        let tile_a = a_base + tile * k;
        let tile_c = c_base + 8 * tile;
        let mut busy = 0;
        for row in 0..m {
            for chunk in 0..chunks {
                busy += load(tile_a + row * (k / 8) + 2 * chunk, tile_a, m * (k / 8));
            }
        }
        for w in 0..n {
            busy += load(b_base + w, b_base, k);
        }
        for row in 0..8 {
            if !first {
                busy += load(tile_c + row, tile_c, m);
            }
            busy += 4 + row_cost(tile_c + row, tile_c, m);
        }
        if !first {
            // Grants before the engine starts: X, W, Y requests within their credits.
            let x_can = (m * chunks).min(m + 8);
            let w_can = n.min(8);
            let (mut x, mut w, mut y, mut rr) = (0, 0, 0, 0);
            while y < 8 {
                let wants = [x < x_can, w < w_can, true];
                let stream = (rr..rr + 4)
                    .map(|s| s % 4)
                    .find(|&s| s < 3 && wants[s as usize])
                    .unwrap();
                match stream {
                    0 => x += 1,
                    1 => w += 1,
                    _ => y += 1,
                }
                rr = (rr + 1) % 4;
            }
            if x == x_can && w == w_can {
                busy += 4;
            }
        }
        total += 28 + busy + 14 + 4 - if tile == 0 { 0 } else { 2 };
    }
    total
}
