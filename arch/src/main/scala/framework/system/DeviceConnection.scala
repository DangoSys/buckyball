package framework.system

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.dlink.DLinkIO
import framework.system.tile.tlink.{Fabric, TLinkIO}
import framework.system.memory.Memory
import framework.system.device.{BootRomLines, Devices}
import memcore.memory.cpu.CpuMemParams

object DeviceConnection {

  def connect(p: SystemParams, dlink: DLinkIO, tilePorts: Seq[TLinkIO]): Unit = {
    val topology = p.topology
    val memory   = p.memory
    val ddr      = p.ddr
    val devices  = p.devices
    val tiles    = topology.tiles
    val hartIds  = tiles.flatMap(_.hartIds)
    require(ddr.clients == 1 && ddr.line == memory.chi)
    val n        = hartIds.size

    val dmaMasters = tiles.size

    val cp = CpuMemParams(memory.chi, tagBits = 6)

    val controlCounts = tiles.map(_.hartIds.size)
    val controlBase   = controlCounts.scanLeft(0)(_ + _)
    val controlCount  = controlCounts.sum

    require(tilePorts.size == tiles.size, "Designed tile count must match software topology")
    val linkedTiles = tilePorts.indices.filter(i => tilePorts(i).t2t.isDefined)
    if (linkedTiles.nonEmpty) {
      val geometries = linkedTiles.map(i => tiles(i).sharedStorage.get.bytes)
      require(geometries.distinct.size == 1, "TLink tiles must share the same public shared-storage geometry")
      val fabric     = Instantiate(new Fabric(ddr.axi, linkedTiles, geometries.head))
      for ((tile, port) <- linkedTiles.zipWithIndex) {
        fabric.io.source(port) <> tilePorts(tile).t2t.get.tx
        tilePorts(tile).t2t.get.rx <> fabric.io.target(port)
      }
    }

    val backing: Instance[Memory] = Instantiate(new Memory(ddr, dmaMasters + 1))
    val chipDevices = Instantiate(new Devices(devices, cp, hartIds))
    chipDevices.io.sources := 0.U
    val external  = Instantiate(new framework.system.memory.UncachedAxi(cp, ddr.axi, n))
    val scuParams = devices.scu
    val scu       = Instantiate(new sims.scu.SystemControl(n, cp, scuParams))
    scu.io.hartIds := VecInit(hartIds.map(_.U(32.W)))
    for (core <- hartIds.indices) {
      val request     = chipDevices.io.externalRequest(core)
      val response    = chipDevices.io.externalResponse(core)
      val inScu       = request.bits.addr >= scuParams.baseAddress.U &&
        request.bits.addr < (scuParams.baseAddress + scuParams.totalSizeBytes).U
      val selectedScu = RegInit(false.B)
      when(request.fire)(selectedScu   := inScu)
      scu.io.request(core).valid       := request.valid && inScu
      scu.io.request(core).bits        := request.bits
      external.io.request(core).valid  := request.valid && !inScu
      external.io.request(core).bits   := request.bits
      request.ready                    := Mux(inScu, scu.io.request(core).ready, external.io.request(core).ready)
      response.valid                   := Mux(selectedScu, scu.io.response(core).valid, external.io.response(core).valid)
      response.bits                    := Mux(selectedScu, scu.io.response(core).bits, external.io.response(core).bits)
      scu.io.response(core).ready      := selectedScu && response.ready
      external.io.response(core).ready := !selectedScu && response.ready
    }
    scu.io.failure := VecInit(tilePorts.flatMap(_.failure))
    backing.io.dma(dmaMasters) <> external.io.axi
    dlink.axi <> backing.io.axi

    val coreBase = tiles.scanLeft(0)(_ + _.hartIds.size)
    for ((tile, index) <- tilePorts.zipWithIndex) {
      tile.tileId := index.U
      for ((port, id) <- tile.hartIds.zip(tiles(index).hartIds)) port           := id.U(64.W)
      for ((port, id) <- tile.executionIds.zip(tiles(index).executionIds)) port := id.U(64.W)
      tile.time := chipDevices.io.time
      for (local <- tiles(index).hartIds.indices) {
        val core = coreBase(index) + local
        tile.resetVector(local) := devices.bootrom.get.base.U
        tile.interrupts(local)  := chipDevices.io.interrupts(core)
        chipDevices.io.request(core) <> tile.uncachedRequest(local)
        tile.uncachedResponse(local) <> chipDevices.io.response(core)
      }
      // Tile-local requester IDs keep homogeneous tiles identical. Only endpoint identity
      // changes at the shared control fabric; clocks, transactions and backpressure do not.
      for (lane  <- 0 until (if (index == 0) 0 else 2 * controlCounts(index))) {
        val localNode   = lane + 1
        val globalIndex = controlBase(index) + lane                      % controlCounts(index) +
          (if (lane >= controlCounts(index)) controlCount else 0)
        val globalNode  = globalIndex + 1
        val source      = tile.control(lane)
        val remoteCount = controlCount - controlCounts.head
        val remoteIndex = controlBase(index) - controlCounts.head + lane % controlCounts(index) +
          (if (lane >= controlCounts(index)) remoteCount else 0)
        val target      = tilePorts.head.remoteControl(remoteIndex)
        target <> source
        target.req.bits.srcId   := globalNode.U
        target.txRsp.bits.srcId := globalNode.U
        target.txDat.bits.srcId := globalNode.U
        source.rxRsp.bits.tgtId := localNode.U
        source.rxDat.bits.tgtId := localNode.U
        when(source.req.valid) {
          assert(source.req.bits.srcId === localNode.U && source.req.bits.returnNid === 0.U)
        }
        when(source.txRsp.valid)(assert(source.txRsp.bits.srcId === localNode.U))
        when(source.txDat.valid)(assert(source.txDat.bits.srcId === localNode.U))
        when(target.rxRsp.valid)(assert(target.rxRsp.bits.tgtId === globalNode.U))
        when(target.rxDat.valid)(assert(target.rxDat.bits.tgtId === globalNode.U))
      }
      if (index == 0) {
        val request  = tile.backingRequest(0)
        val response = tile.backingResponse(0)
        devices.bootrom match {
          case Some(rom) =>
            val lines = Instantiate(new BootRomLines(rom, memory.chi))
            lines.io.request <> request
            response <> lines.io.response
            backing.io.lineRequest(0) <> lines.io.memoryRequest
            lines.io.memoryResponse <> backing.io.lineResponse(0)
          case None      =>
            backing.io.lineRequest(0) <> request
            response <> backing.io.lineResponse(0)
        }
      }
      backing.io.dma(index) <> tile.mem
    }
  }

}
