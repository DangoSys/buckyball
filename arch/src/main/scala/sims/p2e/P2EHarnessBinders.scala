package sims.p2e

import chisel3._
import org.chipsalliance.cde.config.Config

import chipyard.harness.{HarnessBinder, HasHarnessInstantiators}
import chipyard.iobinders.{AXI4MemPort, UARTPort}

class WithP2ETieOffAXIMem
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: AXI4MemPort, chipId: Int) =>
        val axi = port.io.bits
        axi.aw.ready := false.B
        axi.w.ready  := false.B
        axi.b.valid  := false.B
        axi.b.bits   := DontCare
        axi.ar.ready := false.B
        axi.r.valid  := false.B
        axi.r.bits   := DontCare
    })

class WithNoUARTAdapter
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: UARTPort, chipId: Int) =>
        port.io.rxd := true.B
    })

class WithP2EHarness
    extends Config(
      new WithP2ETieOffAXIMem ++
        new WithNoUARTAdapter ++
        new chipyard.harness.WithSimAXIMMIO ++
        new chipyard.harness.WithSerialTLTiedOff ++
        new chipyard.harness.WithTieOffInterrupts ++
        new chipyard.harness.WithTieOffL2FBusAXI ++
        new chipyard.harness.WithTiedOffJTAG ++
        new chipyard.harness.WithTiedOffDMI ++
        new chipyard.harness.WithClockFromHarness ++
        new chipyard.harness.WithResetFromHarness ++
        new chipyard.harness.WithAbsoluteFreqHarnessClockInstantiator ++
        new chipyard.iobinders.WithAXI4MemPunchthrough ++
        new chipyard.iobinders.WithAXI4MMIOPunchthrough ++
        new chipyard.iobinders.WithNMITiedOff
    )
