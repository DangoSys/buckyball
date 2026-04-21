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
        val axi   = port.io.bits
        val shell = th.pegasusShell.io
        // Connect mmio_axi4 to shell's MMIO SCU
        shell.mmio_awid    := axi.aw.bits.id.asTypeOf(UInt(4.W))
        shell.mmio_awaddr  := axi.aw.bits.addr.asTypeOf(UInt(32.W))
        shell.mmio_awlen   := axi.aw.bits.len
        shell.mmio_awsize  := axi.aw.bits.size
        shell.mmio_awburst := axi.aw.bits.burst
        shell.mmio_awvalid := axi.aw.valid
        axi.aw.ready       := shell.mmio_awready

        shell.mmio_wdata  := axi.w.bits.data.asTypeOf(UInt(64.W))
        shell.mmio_wstrb  := axi.w.bits.strb.asTypeOf(UInt(8.W))
        shell.mmio_wlast  := axi.w.bits.last
        shell.mmio_wvalid := axi.w.valid
        axi.w.ready       := shell.mmio_wready

        axi.b.bits.id     := shell.mmio_bid.asTypeOf(axi.b.bits.id)
        axi.b.bits.resp   := shell.mmio_bresp
        axi.b.valid       := shell.mmio_bvalid
        shell.mmio_bready := axi.b.ready

        shell.mmio_arid    := axi.ar.bits.id.asTypeOf(UInt(4.W))
        shell.mmio_araddr  := axi.ar.bits.addr.asTypeOf(UInt(32.W))
        shell.mmio_arlen   := axi.ar.bits.len
        shell.mmio_arsize  := axi.ar.bits.size
        shell.mmio_arburst := axi.ar.bits.burst
        shell.mmio_arvalid := axi.ar.valid
        axi.ar.ready       := shell.mmio_arready

        axi.r.bits.id     := shell.mmio_rid.asTypeOf(axi.r.bits.id)
        axi.r.bits.data   := shell.mmio_rdata.asTypeOf(axi.r.bits.data)
        axi.r.bits.resp   := shell.mmio_rresp
        axi.r.bits.last   := shell.mmio_rlast
        axi.r.valid       := shell.mmio_rvalid
        shell.mmio_rready := axi.r.ready
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
        new chipyard.harness.WithClockFromHarness ++
        new chipyard.harness.WithResetFromHarness ++
        new chipyard.iobinders.WithAXI4MemPunchthrough ++
        new chipyard.iobinders.WithAXI4MMIOPunchthrough ++
        new chipyard.iobinders.WithNMITiedOff
    )
