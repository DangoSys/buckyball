package examples.designs.poly.devices.poly

import scala.jdk.CollectionConverters._

import framework.thread.configs.ThreadParam

import framework.generic.arbiter.{RRArbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.device.configs.DeviceParam
import framework.hart.cbus.CBus
import framework.device.shm.{Params => CommunicationParams, Network, Node}
import framework.hart.cbus.generated.HartComponents
import buckyball.config.Design
import framework.device.sims.configs.GlobalConfig
import framework.hart.hlink.HLinkIO
import framework.hart.memory.home.Home
import framework.generic.ddr.Bridge
import framework.device.peripherals.Controller
import framework.device.peripherals.bootrom.BootRomLines
import framework.device.dlink.BaseDevice
import framework.generic.bus.axi4
import framework.generic.bus.chi.snf.LineResponse
import framework.generic.queue.{Queue => ResponseQueue}

/** Configured harts share the device coherence fabric. */
@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.port) {
  val device = b.device
  val harts  = b.harts

  // The data banks of the harts' and the threads' accelerators: (width, entries).
  val dataBanks = harts.flatMap(_.acc).map(acc => (acc.memDomain.bankWidth, acc.memDomain.bankEntries)) ++
    design.getDevice.getHartsList.asScala.flatMap(source =>
      ThreadParam.ofHart(design, source).flatMap(_.acc).map(acc => (acc.memDomain.bankWidth, acc.memDomain.bankEntries))
    )

  require(dataBanks.forall(_ == (128, 1024)), "Poly data banks must all be 16 KiB")
  require(
    harts.flatMap(_.shm).forall(memory => memory.dataBits == 128 && memory.entriesPerBank == 1024),
    "Poly communication banks must match its data banks"
  )
  require(harts.forall(p => !p.hlink.backing && p.hlink.requesters == 1))
  val sharedHome = harts.head.memory.home
  require(
    harts.forall(_.memory.memoryAccess.memory.chi == sharedHome.memory.chi),
    "Hart coherent channels must match the shared Home"
  )
  require(sharedHome.memory.agents == harts.size)

  val bus: Instance[CBus] =
    Instantiate(new CBus(b.harts, b.device.ddr, CBus.generators(design, b.harts, b.sim, HartComponents.factories)))
  val communication = CommunicationParams(256 * 1024, 16 * 1024, harts.size, device.ddr.axi)
  val links         = Instantiate(new Network(communication, HartLinks(harts.size)))
  require(harts.forall(_.hlink.service))
  for (hart <- harts.indices) {
    val endpoint = Instantiate(new Node(communication))
    endpoint.io.hartId := harts(hart).hartId.U
    val payload = Instantiate(new Payload)
    payload.io.host <> bus.services(hart)
    payload.io.payload <> bus.payloads(hart)
    endpoint.io.request <> payload.io.storage.request
    payload.io.storage.reply <> endpoint.io.reply
    payload.io.storage.busy := endpoint.io.busy
    endpoint.io.local <> payload.io.local
    links.io.harts(hart) <> endpoint.io.link
  }
  val home: Instance[Home] = Instantiate(new Home(sharedHome, false, bus.pages.size))
  val peripherals: Instance[Controller] =
    Instantiate(new Controller(device, device.ddr.axi, harts.head.memory.memoryAccess.memory))

  val peripheralBus: Instance[axi4.Decoder] = Instantiate(new axi4.Decoder(
    Seq(
      (device.clint.base, device.clint.bytes),
      (device.plic.base, device.plic.bytes),
      (device.scu.baseAddress, device.scu.totalSizeBytes),
      (device.trace.base, device.trace.bytes)
    ),
    device.ddr.axi
  ))

  val rom:       Instance[BootRomLines]                = Instantiate(new BootRomLines(device.bootrom, sharedHome.memory.chi))
  val bridge:    Instance[Bridge]                      = Instantiate(new Bridge(device.ddr))
  val responses: Instance[RRArbiter[LineResponse]]     =
    Instantiate(new RRArbiter(new LineResponse(sharedHome.memory.chi), 2))
  val response:  Instance[ResponseQueue[LineResponse]] =
    Instantiate(new ResponseQueue(new LineResponse(sharedHome.memory.chi), 1, true, false))
  val memoryBus: Instance[axi4.Interconnect]           =
    Instantiate(new axi4.Interconnect(axi4.InterconnectParam(device.ddr.port, 2, None)))

  home.io.active            := true.B
  home.io.blockRequesterRsp := false.B
  home.io.requesters <> bus.coherent
  home.page <> bus.pages

  peripheralBus.io.in <> bus.uncached
  peripherals.io.axi <> peripheralBus.io.matched
  bus.external <> peripheralBus.io.other
  peripherals.io.hardwareInterrupt.sources := 0.U
  bus.interrupts                           := peripherals.io.hardwareInterrupt.harts

  require(device.ddr.clients == 1)
  val request = home.io.memoryReq
  val inRom   = request.bits.addr >= device.bootrom.base.U &&
    request.bits.addr < (device.bootrom.base + device.bootrom.bytes).U
  rom.io.request.valid       := request.valid && inRom
  rom.io.request.bits        := request.bits
  bridge.io.request(0).valid := request.valid && !inRom
  bridge.io.request(0).bits  := request.bits
  request.ready              := Mux(inRom, rom.io.request.ready, bridge.io.request(0).ready)

  responses.io.in(0) <> rom.io.response
  responses.io.in(1) <> bridge.io.response(0)

  response.io.enq <> responses.io.out
  home.io.memoryResp <> response.io.deq

  memoryBus.io.in(0) <> bus.dlink.axi
  memoryBus.io.in(1) <> bridge.io.axi
  dlink.axi <> memoryBus.io.out
}
