package framework.system.link

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.tilelink.{TLBundle, TLBundleA, TLBundleD, TLBundleParameters}

/**
 * TLXbarModule - real TileLink crossbar (pure Module, no Diplomacy/LazyModule).
 *
 * Supports N inputs to 1 output for TL-UH (no B/C/E coherence channels):
 *  - Round-robin arbitration on the A channel (multiple masters -> 1 slave)
 *  - Response routing on the D channel using the high bits of the source ID
 *
 * Source ID remapping
 * -------------------
 * Each input's source-ID space is prefixed with the input index so that the
 * downstream slave sees a globally unique source ID, and so that responses on
 * the D channel can be routed back to the originating input by inspecting the
 * top `log2Ceil(nInputs)` bits of `source`.
 *
 * Example with nInputs = 4 and `tlParams.sourceBits = 8`:
 *   Input 0: original source 0..63   -> remapped 0..63    (prefix 00)
 *   Input 1: original source 0..63   -> remapped 64..127  (prefix 01)
 *   Input 2: original source 0..63   -> remapped 128..191 (prefix 10)
 *   Input 3: original source 0..63   -> remapped 192..255 (prefix 11)
 *
 * The original source bit-width inside each input is therefore reduced by
 * `log2Ceil(nInputs)` bits. Callers must size `tlParams.sourceBits` accordingly
 * (i.e. `inputSourceBits + log2Ceil(nInputs)`); this module does not widen the
 * field on its own.
 *
 * Single-input case (nInputs == 1) is a direct pass-through with no
 * arbitration or remap overhead.
 *
 * TL-UH only: B/C/E channels are tied off. When `tlParams.hasBCE = false`
 * those channels are not present on the TLBundle anyway (they degrade to
 * internal `WireDefault`s), so the tie-off below is a safe no-op in that
 * configuration and provides correct behavior if BCE is ever enabled.
 */
@instantiable
class TLXbarModule(
  val nInputs:  Int,
  val tlParams: TLBundleParameters)
    extends Module {

  require(nInputs >= 1, "TLXbar must have at least 1 input")

  @public
  val io = IO(new Bundle {
    val in  = Vec(nInputs, Flipped(new TLBundle(tlParams)))
    val out = new TLBundle(tlParams)
  })

  if (nInputs == 1) {
    // Pass-through: no arbitration, no remapping needed.
    io.out <> io.in(0)
  } else {
    val inputIdxBits   = log2Ceil(nInputs)
    val sourceBits     = tlParams.sourceBits
    require(
      sourceBits > inputIdxBits,
      s"TLXbar: tlParams.sourceBits ($sourceBits) must be greater than " +
        s"log2Ceil(nInputs) ($inputIdxBits) to leave room for the input-index prefix"
    )
    val origSourceBits = sourceBits - inputIdxBits

    // ---------------------------------------------------------------------
    // A channel: round-robin arbitration of `nInputs` masters onto the slave.
    // ---------------------------------------------------------------------
    val arb = Module(new RRArbiter(new TLBundleA(tlParams), nInputs))
    for (i <- 0 until nInputs) {
      arb.io.in(i).valid       := io.in(i).a.valid
      arb.io.in(i).bits        := io.in(i).a.bits
      // Remap the source ID by prefixing the input index in the high bits.
      // The original source field is assumed to occupy the low `origSourceBits`
      // bits of the master's `source`; any high bits above `origSourceBits`
      // are dropped (they should be zero by construction).
      arb.io.in(i).bits.source :=
        Cat(i.U(inputIdxBits.W), io.in(i).a.bits.source(origSourceBits - 1, 0))
      io.in(i).a.ready         := arb.io.in(i).ready
    }
    io.out.a <> arb.io.out

    // ---------------------------------------------------------------------
    // D channel: route the slave's response back to the originating input
    // using the high bits of `source`.
    // ---------------------------------------------------------------------
    val dSource     = io.out.d.bits.source
    val targetInput = dSource(sourceBits - 1, sourceBits - inputIdxBits)

    for (i <- 0 until nInputs) {
      val sel = targetInput === i.U(inputIdxBits.W)
      io.in(i).d.valid       := io.out.d.valid && sel
      io.in(i).d.bits        := io.out.d.bits
      // Restore the original (un-prefixed) source ID for the master.
      // Zero-extend back up to `sourceBits` so the field width matches.
      io.in(i).d.bits.source :=
        dSource(origSourceBits - 1, 0).pad(sourceBits)
    }
    // Backpressure from the selected input feeds back to the slave.
    io.out.d.ready := MuxLookup(targetInput, false.B)(
      (0 until nInputs).map(i => i.U(inputIdxBits.W) -> io.in(i).d.ready)
    )

    // ---------------------------------------------------------------------
    // B/C/E: TL-UH (no coherence). Tie off in both directions defensively.
    // For `hasBCE = false` these accessors return internal `WireDefault`s,
    // so the assignments below have no observable effect on the IO. They
    // become meaningful only if a future caller enables BCE.
    // ---------------------------------------------------------------------
    if (tlParams.hasBCE) {
      // Slave -> Master: no B requests, accept any C release immediately.
      io.out.b.ready := true.B
      io.out.c.valid := false.B
      io.out.c.bits  := DontCare
      io.out.e.valid := false.B
      io.out.e.bits  := DontCare

      for (i <- 0 until nInputs) {
        io.in(i).b.valid := false.B
        io.in(i).b.bits  := DontCare
        io.in(i).c.ready := true.B
        io.in(i).e.ready := true.B
      }
    }
  }
}
