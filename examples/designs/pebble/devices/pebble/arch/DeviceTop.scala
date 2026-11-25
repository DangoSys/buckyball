package examples.designs.pebble.devices.pebble

import framework.generic.arbiter.{RRArbiter}

import chisel3._
import chisel3.util._
import framework.device.peripherals.bootrom.BootRomLines
import framework.device.peripherals.Controller
import framework.generic.bus.axi4.Decoder
import framework.generic.bus.chi.snf.LineResponse
import framework.generic.queue.{Queue => ResponseQueue}
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.device.configs.DeviceParam
import framework.hart.cbus.CBus
import framework.hart.cbus.generated.HartComponents
import buckyball.config.Design
import framework.device.sims.configs.GlobalConfig
import framework.hart.hlink.HLinkIO
import framework.device.dlink.BaseDevice
import framework.generic.contract.sims.configs.SimParam

@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.axi) {
  val devicep = b.device

  val Seq(hartp) = b.harts
  require(hartp.mtimeAddress == devicep.clint.base + 0xbff8, "Hart mtime address must match CLINT")

  val bus =
    Instantiate(new CBus(b.harts, b.device.ddr, CBus.generators(design, b.harts, b.sim, HartComponents.factories)))
  dlink <> bus.dlink

  val peripherals =
    Instantiate(new Controller(devicep, devicep.ddr.axi, hartp.memory.memoryAccess.memory))

  val peripheralBus = Instantiate(new Decoder(
    Seq(
      (devicep.clint.base, devicep.clint.bytes),
      (devicep.plic.base, devicep.plic.bytes),
      (devicep.scu.baseAddress, devicep.scu.totalSizeBytes),
      (devicep.trace.base, devicep.trace.bytes)
    ),
    devicep.ddr.axi
  ))

  peripheralBus.io.in <> bus.uncached
  peripherals.io.axi <> peripheralBus.io.matched
  bus.external <> peripheralBus.io.other
  peripherals.io.hardwareInterrupt.sources := 0.U
  bus.interrupts                           := peripherals.io.hardwareInterrupt.harts

  val romParams = devicep.bootrom
  val rom       = Instantiate(new BootRomLines(devicep.bootrom, hartp.memory.home.memory.chi))

  val backing = bus.backing(0)
  val ddr     = bus.memory(0)
  val inRom   = backing.req.bits.addr >= romParams.base.U &&
    backing.req.bits.addr < (romParams.base + romParams.bytes).U
  rom.io.request.valid := backing.req.valid && inRom
  rom.io.request.bits  := backing.req.bits
  ddr.req.valid        := backing.req.valid && !inRom
  ddr.req.bits         := backing.req.bits
  backing.req.ready    := Mux(inRom, rom.io.request.ready, ddr.req.ready)

  val responses = Instantiate(new RRArbiter(new LineResponse(hartp.memory.home.memory.chi), 2))
  responses.io.in(0) <> rom.io.response
  responses.io.in(1) <> ddr.resp

  // Buffer the selected answer so downstream backpressure cannot change its source.
  val response = Instantiate(new ResponseQueue(new LineResponse(hartp.memory.home.memory.chi), 1, false, false))

  response.io.enq <> responses.io.out
  backing.resp <> response.io.deq

}
