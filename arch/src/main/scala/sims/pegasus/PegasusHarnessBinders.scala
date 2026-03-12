package sims.pegasus

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Config}

import chipyard.harness._
import chipyard.iobinders._

import pegasus._

// HarnessBinder: connect ChipTop ExtMem AXI4 port to PegasusShell chip_mem interface
// Replaces WithBlackBoxSimMem (Verilator simulation DRAM model)
class WithPegasusAXIMem
    extends HarnessBinder({
      case (th: PegasusHarness, port: AXI4MemPort, chipId: Int) => {
        th.connectChipMem(port)
      }
    })

// HarnessBinder: connect ChipTop UART TX to PegasusShell (goes to UARTCapture)
// Replaces WithUARTAdapter (Verilator stdout adapter)
class WithPegasusUART
    extends HarnessBinder({
      case (th: PegasusHarness, port: UARTPort, chipId: Int) => {
        th.pegasusShell.io.uart_tx := port.io.txd
        port.io.rxd                := true.B // UART RX idle high (no input from host)
      }
    })

// Tie off JTAG (not used on FPGA; debug via GDB/OpenOCD over UART is possible but not in scope)
class WithPegasusTiedOffJTAG
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: JTAGPort, chipId: Int) => {
        port.io.TCK := true.B.asClock
        port.io.TMS := true.B
        port.io.TDI := true.B
      }
    })

// Tie off DMI (no simulation-side debug model needed on FPGA)
class WithPegasusTiedOffDMI
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: DMIPort, chipId: Int) => {
        port.io.dmi.req.valid  := false.B
        port.io.dmi.req.bits   := DontCare
        port.io.dmi.resp.ready := true.B
        port.io.dmiClock       := false.B.asClock
        port.io.dmiReset       := true.B
      }
    })

// Aggregate config fragment: select PegasusHarness binders
// and override Verilator-only simulation models
class WithPegasusHarness
    extends Config(
      new WithPegasusAXIMem ++
        new WithPegasusUART ++
        new WithPegasusTiedOffJTAG ++
        new WithPegasusTiedOffDMI ++
        // Standard binders that are safe to keep on FPGA
        new chipyard.harness.WithTieOffInterrupts ++
        new chipyard.harness.WithTieOffL2FBusAXI ++
        new chipyard.harness.WithGPIOTiedOff ++
        new chipyard.harness.WithGPIOPinsTiedOff ++
        new chipyard.harness.WithDriveChipIdPin ++
        new chipyard.harness.WithCustomBootPinPlusArg ++
        new chipyard.harness.WithSerialTLTiedOff ++
        new chipyard.harness.WithClockFromHarness ++
        new chipyard.harness.WithResetFromHarness ++
        new chipyard.harness.WithAbsoluteFreqHarnessClockInstantiator
    )
