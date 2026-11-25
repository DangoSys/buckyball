package examples.designs.gemmini.devices.gemmini

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, Instantiate}
import framework.device.peripherals.Controller
import framework.generic.bus.axi4.Decoder
import framework.hart.cbus.CBus
import framework.hart.cbus.generated.HartComponents
import buckyball.config.Design
import framework.device.sims.configs.GlobalConfig
import framework.device.dlink.BaseDevice

@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.axi) {
  val deviceParam = b.device

  val Seq(hartParam) = b.harts

  val bus =
    Instantiate(new CBus(b.harts, b.device.ddr, CBus.generators(design, b.harts, b.sim, HartComponents.factories)))
  dlink <> bus.dlink

  val peripherals =
    Instantiate(new Controller(deviceParam, deviceParam.ddr.axi, hartParam.memory.memoryAccess.memory))

  val peripheralBus = Instantiate(new Decoder(
    Seq(
      (deviceParam.clint.base, deviceParam.clint.bytes),
      (deviceParam.plic.base, deviceParam.plic.bytes),
      (deviceParam.scu.baseAddress, deviceParam.scu.totalSizeBytes),
      (deviceParam.trace.base, deviceParam.trace.bytes)
    ),
    deviceParam.ddr.axi
  ))

  peripheralBus.io.in <> bus.uncached
  peripherals.io.axi <> peripheralBus.io.matched
  bus.external <> peripheralBus.io.other
  peripherals.io.hardwareInterrupt.sources := 0.U
  bus.interrupts                           := peripherals.io.hardwareInterrupt.harts

}
