package sims.pegasus

import chisel3._

import org.chipsalliance.cde.config.Config

import chipyard.harness.{HarnessBinder, HasHarnessInstantiators}
import chipyard.iobinders.{AXI4MMIOPort, AXI4MemPort, DMIPort, JTAGPort, UARTPort}

class WithPegasusAXIMem
    extends HarnessBinder({
      case (th: PegasusHarness, port: AXI4MemPort, chipId: Int) =>
        th.connectChipMem(port)
    })

class WithPegasusUART
    extends HarnessBinder({
      case (th: PegasusHarness, port: UARTPort, chipId: Int) =>
        th.pegasusShell.io.uart_tx := port.io.txd
        port.io.rxd                := true.B
    })

class WithPegasusAXIMMIO
    extends HarnessBinder({
      case (th: PegasusHarness, port: AXI4MMIOPort, chipId: Int) =>
        withClockAndReset(port.io.clock, th.reset) {
          val idBits = port.io.bits.aw.bits.id.getWidth
          val bId    = RegInit(0.U(idBits.W))
          val bValid = RegInit(false.B)

          port.io.bits.aw.ready := !bValid
          port.io.bits.w.ready  := !bValid
          when(!bValid && port.io.bits.aw.valid && port.io.bits.w.valid) {
            bId    := port.io.bits.aw.bits.id
            bValid := true.B
          }
          when(port.io.bits.b.valid && port.io.bits.b.ready) {
            bValid := false.B
          }

          port.io.bits.b.valid     := bValid
          port.io.bits.b.bits.id   := bId
          port.io.bits.b.bits.resp := 0.U

          port.io.bits.ar.ready    := true.B
          port.io.bits.r.valid     := RegNext(port.io.bits.ar.valid, false.B)
          port.io.bits.r.bits.id   := RegNext(port.io.bits.ar.bits.id)
          port.io.bits.r.bits.data := 0.U
          port.io.bits.r.bits.resp := 0.U
          port.io.bits.r.bits.last := true.B
          port.io.bits.r.bits.user := 0.U.asTypeOf(port.io.bits.r.bits.user)
        }
    })

class WithPegasusTiedOffJTAG
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: JTAGPort, chipId: Int) =>
        port.io.TCK := true.B.asClock
        port.io.TMS := true.B
        port.io.TDI := true.B
    })

class WithPegasusTiedOffDMI
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: DMIPort, chipId: Int) =>
        port.io.dmi.req.valid  := false.B
        port.io.dmi.req.bits   := DontCare
        port.io.dmi.resp.ready := true.B
        port.io.dmiClock       := false.B.asClock
        port.io.dmiReset       := true.B
    })

class WithPegasusHarness
    extends Config(
      new WithPegasusAXIMem ++
        new WithPegasusAXIMMIO ++
        new WithPegasusUART ++
        new WithPegasusTiedOffJTAG ++
        new WithPegasusTiedOffDMI ++
        new chipyard.harness.WithSerialTLTiedOff ++
        new chipyard.harness.WithTieOffInterrupts ++
        new chipyard.harness.WithGPIOTiedOff ++
        new chipyard.harness.WithTieOffL2FBusAXI ++
        new chipyard.harness.WithClockFromHarness ++
        new chipyard.harness.WithResetFromHarness ++
        new chipyard.iobinders.WithAXI4MemPunchthrough ++
        new chipyard.iobinders.WithAXI4MMIOPunchthrough ++
        new chipyard.iobinders.WithNMITiedOff
    )
