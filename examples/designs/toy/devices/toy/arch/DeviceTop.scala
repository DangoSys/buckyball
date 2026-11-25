package examples.designs.toy.devices.toy

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

@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.axi) {
  val deviceParam = b.device

  val Seq(hartParam) = b.harts

  val bus =
    Instantiate(new CBus(b.harts, b.device.ddr, CBus.generators(design, b.harts, b.sim, HartComponents.factories)))
  dlink <> bus.dlink

  val p           = deviceParam
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

  val romParams = deviceParam.bootrom
  val rom       = Instantiate(new BootRomLines(deviceParam.bootrom, hartParam.memory.home.memory.chi))

  val backing = bus.backing(0)
  val ddr     = bus.memory(0)
  val inRom   = backing.req.bits.addr >= romParams.base.U &&
    backing.req.bits.addr < (romParams.base + romParams.bytes).U
  rom.io.request.valid := backing.req.valid && inRom
  rom.io.request.bits  := backing.req.bits
  ddr.req.valid        := backing.req.valid && !inRom
  ddr.req.bits         := backing.req.bits
  backing.req.ready    := Mux(inRom, rom.io.request.ready, ddr.req.ready)

  val responses = Instantiate(new RRArbiter(new LineResponse(hartParam.memory.home.memory.chi), 2))
  responses.io.in(0) <> rom.io.response
  responses.io.in(1) <> ddr.resp

  // Buffer the selected answer so downstream backpressure cannot change its source.
  val response = Instantiate(new ResponseQueue(new LineResponse(hartParam.memory.home.memory.chi), 1, false, false))

  response.io.enq <> responses.io.out
  backing.resp <> response.io.deq

}
