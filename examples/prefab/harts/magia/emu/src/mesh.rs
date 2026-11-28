use bebop_bemu::riscv::{registers::CustomInstruction, Trap};
use bebop_bemu::GuestAccess;
use bebop_bemu::Rocc;

/// The mesh of 4 x 4 tiles behind Loom's custom-0. The model is of the mesh
/// running the GEMM service of the hart's stack (stack/firmware/service.c):
/// a launch does the job the host left in L2, it does not run the tiles'
/// instructions.
pub struct Mesh {
    /// Where the mesh's windows of L2 show the DDR: W, X, Z.
    windows: [u64; 3],
    /// When Loom took the last instruction.
    accepted: u64,
    /// When the launched job completes.
    complete: u64,
    launched: bool,
    /// The tiles' aggregate exit code of the last job.
    result: u32,
}

const TILES: u64 = 16;
/// The entry of the firmware and its job, in L2 (stack/firmware/gemm.h).
const ENTRY: u64 = 0xcc00_0080;
const JOB: u64 = 0xcc00_f000;
/// A tile's L1 after the words its runtime keeps.
const L1_BYTES: u64 = 0xe_0000 - 0x1_2048;

/// Launch to completion of a job: the reset and boot of the tiles, then every busy tile copies
/// X, its columns of W and of Z in and its columns of Z out, each a 2-D iDMA transfer of rows,
/// and runs RedMulE. All tiles share the L2 port: the copies serialize, a row at a time, and
/// the time goes to the rows (W has a row per term), not to the bytes. Fitted to RTL on
/// Verilator, from the host's launch to its last poll: 1x16x64 37k, 4x16x64 41k, 4x768x64 371k,
/// 1x1024x64 456k cycles; the model is within 2 % of each.
const BOOT_CYCLES: u64 = 30_000;
const ROW_CYCLES: u64 = 23;
const DMA_BYTES_PER_CYCLE: u64 = 4;
const FMAS_PER_CYCLE: u64 = 16;

fn words(bytes: u64) -> u64 {
    (bytes + 3) & !3
}

/// Two to the `exponent`, a normal double.
fn pow2(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn widen(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let fraction = f64::from(bits & 0x3ff);
    sign * match (bits >> 10) & 0x1f {
        0 => fraction * pow2(-24),
        31 if fraction == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        exponent => (1024.0 + fraction) * pow2(i32::from(exponent) - 25),
    }
}

/// Rounds to nearest even.
fn narrow(value: f64) -> u16 {
    if value.is_nan() {
        return 0x7e00;
    }
    let sign = ((value.to_bits() >> 48) & 0x8000) as u16;
    let magnitude = value.abs();
    if magnitude >= 65520.0 {
        return sign | 0x7c00;
    }
    // The exponent of the result; subnormals keep that of the smallest normal.
    let exponent = (((value.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-14);
    // Below 2^52 a double plus 2^52 has no fraction bits left: the sum rounds to the nearest even integer.
    let rounded = ((magnitude * pow2(10 - exponent) + pow2(52)) - pow2(52)) as u32;
    if rounded < 1024 {
        return sign | rounded as u16;
    }
    sign | ((((exponent + 15) as u32) << 10) + rounded - 1024) as u16
}

/// One output row, with the same binary16 rounding after every term as the tiles.
fn product_row(xs: &[u16], weights: &[u16], sums: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512fp16") && std::is_x86_feature_detected!("avx512bw") {
        // Both instruction sets used by the kernel are available on this host.
        unsafe { product_row_fp16(xs, weights, sums) };
        return;
    }
    for (&x, row) in xs.iter().zip(weights.chunks_exact(sums.len())) {
        let x = widen(x);
        for (sum, &weight) in sums.iter_mut().zip(row) {
            *sum = narrow(x * widen(weight) + widen(*sum));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512fp16,avx512bw")]
unsafe fn product_row_fp16(xs: &[u16], weights: &[u16], sums: &mut [u16]) {
    use std::arch::x86_64::*;
    let cols = sums.len();
    for (block, out) in sums.chunks_mut(32).enumerate() {
        let mask = u32::MAX >> (32 - out.len());
        // The mask restricts every load and store to the columns of this block.
        let mut sum = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, out.as_ptr().cast()));
        for (&x, row) in xs.iter().zip(weights.chunks_exact(cols)) {
            let x = _mm512_castsi512_ph(_mm512_set1_epi16(x as i16));
            let w = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(
                mask,
                row[block * 32..].as_ptr().cast(),
            ));
            sum = _mm512_fmadd_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                x, w, sum,
            );
        }
        // A NaN remains a NaN for every subsequent FMA. Canonicalize it when we store the result.
        let nan = _mm512_cmp_ph_mask::<_CMP_UNORD_Q>(sum, sum);
        let bits = _mm512_mask_set1_epi16(_mm512_castph_si512(sum), nan, 0x7e00);
        _mm512_mask_storeu_epi16(out.as_mut_ptr().cast(), mask, bits);
    }
}

impl Mesh {
    pub fn new() -> Self {
        Mesh {
            windows: [0xd000_0000, 0xf000_0000, 0xf800_0000],
            accepted: 0,
            complete: 0,
            launched: false,
            result: 0,
        }
    }

    /// The DDR address the mesh reaches at `address` of L2: the firmware and
    /// its data where they are, the windows where Loom pointed them.
    fn ddr(&self, address: u64) -> u64 {
        match address >> 27 {
            0x1a..=0x1d => self.windows[0] + address - 0xd000_0000,
            0x1e => self.windows[1] + address - 0xf000_0000,
            0x1f => self.windows[2] + address - 0xf800_0000,
            _ => address,
        }
    }

    /// Z = Z + X * W in FP16, one fused multiply-add a product, as every tile
    /// computes its columns of the job, or its whole products of a batch; the
    /// result is nonzero when a tile cannot hold its share.
    #[inline(never)]
    fn gemm(&mut self, guest: &mut GuestAccess<'_>) -> u64 {
        let mut job = [0u8; 32];
        guest.read_physical(JOB, &mut job);
        let field = |index: usize| {
            u64::from(u32::from_le_bytes(
                job[index * 4..index * 4 + 4].try_into().unwrap(),
            ))
        };
        let (x, w, z) = (self.ddr(field(0)), self.ddr(field(1)), self.ddr(field(2)));
        let (m, n, k, pitch, batch) = (field(3), field(4), field(5), field(6), field(7));
        // A tile takes at least two columns; the tile with the most. A tile
        // of a batch takes every column of its products.
        assert!(k >= 2, "a job of the mesh has at least two columns");
        assert!(
            batch == 1 || pitch == k,
            "the products of a batch are whole matrices"
        );
        let busy = if batch > 1 {
            batch.min(TILES)
        } else {
            (k / 2).min(TILES)
        };
        let columns = if batch > 1 {
            k
        } else {
            (0..busy)
                .map(|tile| k * (tile + 1) / busy - k * tile / busy)
                .max()
                .unwrap()
        };
        // An odd share is a column wider in L1 (RedMulE takes pairs).
        let even = columns + (columns & 1);
        self.result = u32::from(words(m * n * 2) + words(n * even * 2) + m * even * 2 > L1_BYTES);
        if self.result != 0 {
            return BOOT_CYCLES;
        }
        let read =
            |guest: &mut GuestAccess<'_>, address: u64, rows: u64, cols: u64, stride: u64| {
                let mut bytes = vec![0u8; (rows * cols * 2) as usize];
                if cols == stride {
                    guest.read_physical(address, &mut bytes);
                } else {
                    for (row, output) in bytes.chunks_exact_mut((cols * 2) as usize).enumerate() {
                        guest.read_physical(address + row as u64 * stride * 2, output);
                    }
                }
                bytes
                    .chunks_exact(2)
                    .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                    .collect::<Vec<u16>>()
            };
        for index in 0..batch {
            // W and Z: the K columns of the job in rows `pitch` columns apart.
            let (x, w, z) = (
                x + index * m * n * 2,
                w + index * n * k * 2,
                z + index * m * k * 2,
            );
            let xs = read(guest, x, m, n, n);
            let ws = read(guest, w, n, k, pitch);
            let mut zs = read(guest, z, m, k, pitch);
            let (m, n, k, pitch) = (m as usize, n as usize, k as usize, pitch as usize);
            for (row, sums) in xs.chunks_exact(n).zip(zs.chunks_exact_mut(k)) {
                product_row(row, &ws, sums);
            }
            // Only the columns of the job are written.
            for row in 0..m {
                let bytes: Vec<u8> = zs[row * k..row * k + k]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                guest.write_physical(z + (row * pitch * 2) as u64, &bytes);
            }
        }
        // The copies of every busy tile's share go through the one L2 port; a
        // tile of a batch copies and computes its products one after another.
        let shares = if batch > 1 { batch } else { busy };
        let rounds = (batch + TILES - 1) / TILES;
        BOOT_CYCLES
            + shares * (3 * m + n) * ROW_CYCLES
            + shares * 2 * (m * n + n * columns + 2 * m * columns) / DMA_BYTES_PER_CYCLE
            + rounds * m * n * columns / FMAS_PER_CYCLE
    }
}

impl Rocc for Mesh {
    fn cycles(&self) -> (u64, u64) {
        (self.accepted, self.accepted)
    }
    fn custom(
        &mut self,
        _guest: GuestAccess<'_>,
        request: CustomInstruction,
        _cycle: u64,
    ) -> Result<u64, Trap> {
        Err(Trap::illegal(request.instruction))
    }
    #[inline]
    fn read_register0(&mut self, request: CustomInstruction, cycle: u64) -> Option<u64> {
        let value = match request.instruction >> 25 {
            1 if self.launched && cycle >= self.complete => u64::from(self.result) << 32 | 1,
            1 | 2 => 0,
            _ => return None,
        };
        self.accepted = cycle;
        Some(value)
    }
    #[inline]
    fn custom0(
        &mut self,
        mut guest: GuestAccess<'_>,
        request: CustomInstruction,
        cycle: u64,
    ) -> Result<u64, Trap> {
        if let Some(value) = self.read_register0(request, cycle) {
            return Ok(value);
        }
        self.accepted = cycle;
        match request.instruction >> 25 {
            // Launch: Loom takes it once the mesh is idle.
            0 => {
                assert_eq!(
                    request.rs1, ENTRY,
                    "the mesh model runs the GEMM service only"
                );
                self.accepted = cycle.max(self.complete);
                self.complete = self.accepted + self.gemm(&mut guest);
                self.launched = true;
                Ok(0)
            }
            // A window of the mesh's view of the DDR, while the mesh is idle.
            3 => {
                assert!(
                    request.rs2 < 3 && request.rs1 & 0xfff == 0,
                    "invalid MAGIA mesh window {} at {:#x}",
                    request.rs2,
                    request.rs1
                );
                self.accepted = cycle.max(self.complete);
                self.windows[request.rs2 as usize] = request.rs1;
                Ok(0)
            }
            _ => Err(Trap::illegal(request.instruction)),
        }
    }
    fn barrier(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_rounds_each_half_fma_and_preserves_masked_edges() {
        let mut state = 11u32;
        let mut bits = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        let special = [
            0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7d01,
        ];
        for cols in [2, 15, 31, 32, 33, 64, 71] {
            for terms in [1, 2, 17] {
                for &first in &special {
                    let mut x: Vec<_> = (0..terms).map(|_| bits()).collect();
                    x[0] = first;
                    let w: Vec<_> = (0..terms * cols).map(|_| bits()).collect();
                    let mut out: Vec<_> = (0..cols).map(|_| bits()).collect();
                    let mut expected = out.clone();
                    for (&x, w) in x.iter().zip(w.chunks_exact(cols)) {
                        for (sum, &w) in expected.iter_mut().zip(w) {
                            *sum = narrow(widen(x) * widen(w) + widen(*sum));
                        }
                    }
                    product_row(&x, &w, &mut out);
                    assert_eq!(out, expected);
                }
            }
        }
    }
}
