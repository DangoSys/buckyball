package sims.verilator

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.WithDefaultMMIOPort

import chipyard.harness.{HarnessBinder, HasHarnessInstantiators}
import chipyard.iobinders.{AXI4MMIOPort, UARTPort}

// =============================================================================
// WithBBSimMMIO: wire AXI4 MMIO port.
//
// All registers run in port.io.clock domain (= harness clock = 1 GHz, thanks
// to WithUniformBusFrequencies(1000)).  C++ samples three stable register
// outputs each posedge:
//   - firePulse  = RegNext(wFire)       — 1-cycle pulse, no debounce needed
//   - latchedAddr = Reg latched on AW   — stable address
//   - latchedData = RegEnable on wFire  — stable write data
// bPending is set on wFire (not delayed) so the B response reaches the CPU
// one cycle before C++ processes the event.
// =============================================================================
class WithBBSimMMIO
    extends HarnessBinder({
      case (th: BBSimHarness, port: AXI4MMIOPort, chipId: Int) => {
        withClockAndReset(port.io.clock, th.reset) {
          val addrBits = port.io.bits.aw.bits.addr.getWidth
          val idBits   = port.io.bits.aw.bits.id.getWidth

          val sIdle :: sGotAW :: Nil = Enum(2)
          val state                  = RegInit(sIdle)
          val latchedAddr            = Reg(UInt(addrBits.W))
          val latchedId              = Reg(UInt(idBits.W))
          val bPending               = RegInit(false.B)
          val bId                    = Reg(UInt(idBits.W))

          // --- AW channel ---
          port.io.bits.aw.ready := (state === sIdle)
          when(state === sIdle && port.io.bits.aw.valid) {
            latchedAddr := port.io.bits.aw.bits.addr
            latchedId   := port.io.bits.aw.bits.id
            state       := sGotAW
          }

          // --- W channel ---
          port.io.bits.w.ready := (state === sGotAW)
          val wFire       = (state === sGotAW) && port.io.bits.w.valid
          val latchedData = RegEnable(port.io.bits.w.bits.data, wFire)
          when(wFire) {
            state    := sIdle
            bPending := true.B
            bId      := latchedId
          }

          // --- B channel ---
          when(port.io.bits.b.valid && port.io.bits.b.ready) {
            bPending := false.B
          }
          port.io.bits.b.valid     := bPending
          port.io.bits.b.bits.id   := bId
          port.io.bits.b.bits.resp := 0.U

          // --- Fire pulse for C++ mmio_tick() ---
          // RegNext delays 1 cycle; addr/data are stable registers at that point.
          val firePulse = RegNext(wFire, false.B)
          th.io.mmio_fire      := firePulse
          th.io.mmio_fire_addr := latchedAddr
          th.io.mmio_fire_data := latchedData

          // --- AR channel: accept immediately, return 0 next cycle ---
          port.io.bits.ar.ready := true.B
          val rValid = RegNext(port.io.bits.ar.valid, false.B)
          val rId    = RegNext(port.io.bits.ar.bits.id)
          port.io.bits.r.valid     := rValid
          port.io.bits.r.bits.data := 0.U
          port.io.bits.r.bits.resp := 0.U
          port.io.bits.r.bits.last := true.B
          port.io.bits.r.bits.id   := rId
        }
      }
    })

// =============================================================================
// WithNoUARTAdapter: suppress UARTAdapter; tie RX high (idle line)
// =============================================================================
class WithNoUARTAdapter
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: UARTPort, chipId: Int) => {
        port.io.rxd := true.B
      }
    })

// =============================================================================
// BBSimConfig
// =============================================================================
class BBSimConfig
    extends Config(
      new WithNoUARTAdapter ++
        new WithBBSimMMIO ++
        new WithDefaultMMIOPort ++
        new chipyard.config.WithUniformBusFrequencies(1000.0) ++ // match harness 1 GHz so MMIO clock = harness clock
        new chipyard.harness.WithBlackBoxSimMem ++
        new chipyard.harness.WithSerialTLTiedOff ++
        new chipyard.harness.WithTieOffInterrupts ++
        new chipyard.harness.WithGPIOTiedOff ++
        new chipyard.harness.WithTieOffL2FBusAXI ++
        new chipyard.harness.WithClockFromHarness ++
        new chipyard.harness.WithResetFromHarness ++
        new chipyard.harness.WithAbsoluteFreqHarnessClockInstantiator ++
        new chipyard.iobinders.WithAXI4MemPunchthrough ++
        new chipyard.iobinders.WithAXI4MMIOPunchthrough ++
        new chipyard.iobinders.WithNMITiedOff
    )

class BuckyballToyBBSimConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.toy.BuckyballToyConfig
    )

// =============================================================================
// BBSimHarness
// =============================================================================
class BBSimHarness(implicit val p: Parameters) extends Module with HasHarnessInstantiators {

  val io = IO(new Bundle {
    val mmio_fire      = Output(Bool())
    val mmio_fire_addr = Output(UInt(32.W))
    val mmio_fire_data = Output(UInt(64.W))
  })

  // Defaults; WithBBSimMMIO binder overrides these.
  io.mmio_fire      := false.B
  io.mmio_fire_addr := 0.U
  io.mmio_fire_data := 0.U

  def referenceClockFreqMHz: Double = 1000.0
  def referenceClock:        Clock  = clock
  def referenceReset:        Reset  = reset

  val success = WireInit(false.B)

  val lazyDuts = instantiateChipTops()
}
