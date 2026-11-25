package framework.ball.bbus.channels

import framework.generic.contract.ball.BallHostParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite
import framework.generic.contract.memdomain.BankPort

/**
 * Channel allocation and release
 * Connects memory clients to the layout table
 *
 * Configured Ball read and write ports retain their channels until completion.
 * The port ordering is determined by the Ball connection configuration.
 * Requests retain their channel until the owning Ball completes and responses drain.
 */
@instantiable
class Channels(host: BallHostParam, bp: BallDomainParam) extends Module {
  // Each configured Ball owns its declared read and write ports.
  val ownerCount = bp.balls.size
  val totalRead  = bp.balls.map(_.inBW).sum
  val totalWrite = bp.balls.map(_.outBW).sum
  val link       = host.link

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId   = Input(UInt(64.W))
    // Ball read and write ports in configuration order.
    val bankRead  = Vec(totalRead, new BankRead(link))
    val bankWrite = Vec(totalWrite, new BankWrite(link))
    val active    = Input(Vec(bp.balls.size, Bool()))

    val instructionIds = Input(Vec(ownerCount, UInt(64.W)))

    // Output to the layout table
    val mem_req = Vec(host.bankRequestPorts, new BankPort(host.link))
  })

  val readPortOwner  = bp.balls.zipWithIndex.flatMap { case (ball, owner) => Seq.fill(ball.inBW)(owner) }
  val writePortOwner = bp.balls.zipWithIndex.flatMap { case (ball, owner) => Seq.fill(ball.outBW)(owner) }

  // -----------------------------------------------------------------------------
  // Mapping table for configured Ball request ports
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid  = Bool()
    val isRead = Bool()
    val id     = UInt(log2Ceil(math.max(totalRead, totalWrite)).W)
    val owner  = UInt(math.max(1, log2Ceil(ownerCount)).W)
  }

  val mappingTable = RegInit(VecInit(
    Seq.fill(host.bankRequestPorts)(0.U.asTypeOf(new MappingTableEntry))
  ))

  // A route stays assigned until its Ball releases it, so maintain the reverse ownership at allocation time. This keeps
  // readiness and request detection from repeatedly scanning every backend
  // request port for the same port.
  val portSlots          = math.max(totalRead, totalWrite)
  val readPortAllocated  = RegInit(VecInit(Seq.fill(portSlots)(false.B)))
  val writePortAllocated = RegInit(VecInit(Seq.fill(portSlots)(false.B)))
  val channelRows        = RegInit(VecInit(Seq.fill(host.bankRequestPorts)(0.U(3.W))))

  // Allocate or release one channel per cycle, retaining its owner through response drain.
  for (i <- 0 until totalRead) {
    io.bankRead(i).io.req.ready  := false.B
    io.bankRead(i).io.resp.valid := false.B
    io.bankRead(i).io.resp.bits  := DontCare

  }

  for (i <- 0 until totalWrite) {
    io.bankWrite(i).io.req.ready  := false.B
    io.bankWrite(i).io.resp.valid := false.B
    io.bankWrite(i).io.resp.bits  := DontCare
  }

  val requestedReads  = VecInit((0 until totalRead).map(i => io.active(readPortOwner(i)) && !readPortAllocated(i)))
  val requestedWrites = VecInit((0 until totalWrite).map(i => io.active(writePortOwner(i)) && !writePortAllocated(i)))

  val requested = VecInit(requestedReads ++ requestedWrites)
  val freePorts = mappingTable.map(entry => !entry.valid)

  // Connect mapped entries to layout request ports
  for (i <- 0 until host.bankRequestPorts) {
    io.mem_req(i).read.req.valid   := false.B
    io.mem_req(i).read.req.bits    := DontCare
    io.mem_req(i).read.resp.ready  := false.B
    io.mem_req(i).write.req.valid  := false.B
    io.mem_req(i).write.req.bits   := DontCare
    io.mem_req(i).write.resp.ready := false.B
    io.mem_req(i).bank_id          := 0.U
    io.mem_req(i).group_id         := 0.U
    io.mem_req(i).is_shared        := false.B
    io.mem_req(i).thread_id        := io.ownerId
    io.mem_req(i).rob_id           := 0.U
    io.mem_req(i).inst_id          := 0.U

    val isRead    = mappingTable(i).isRead
    val rid       = mappingTable(i).id
    val wid       = mappingTable(i).id
    val ballRead  = io.bankRead(rid).io
    val ballWrite = io.bankWrite(wid).io
    val rbank_id  = io.bankRead(rid).bank_id
    val wbank_id  = io.bankWrite(wid).bank_id
    val rgroup_id = io.bankRead(rid).group_id
    val wgroup_id = io.bankWrite(wid).group_id
    val rrob_id   = io.bankRead(rid).rob_id
    val wrob_id   = io.bankWrite(wid).rob_id

    when(mappingTable(i).valid) {
      when(isRead) {
        io.mem_req(i).read <> ballRead
        io.mem_req(i).bank_id  := rbank_id
        io.mem_req(i).group_id := rgroup_id
        io.mem_req(i).rob_id   := rrob_id
        io.mem_req(i).inst_id  := io.instructionIds(mappingTable(i).owner)
      }.otherwise {
        io.mem_req(i).write <> ballWrite
        io.mem_req(i).bank_id  := wbank_id
        io.mem_req(i).group_id := wgroup_id
        io.mem_req(i).rob_id   := wrob_id
        io.mem_req(i).inst_id  := io.instructionIds(mappingTable(i).owner)
      }
    }
  }

  for (i <- 0 until host.bankRequestPorts) {
    val issued    = io.mem_req(i).read.req.fire || io.mem_req(i).write.req.fire
    val completed = io.mem_req(i).read.resp.fire || io.mem_req(i).write.resp.fire
    when(issued =/= completed) {
      channelRows(i) := Mux(issued, channelRows(i) + 1.U, channelRows(i) - 1.U)
    }
  }

  val release = VecInit((0 until host.bankRequestPorts).map { i =>
    val ownerFinished = !io.active(mappingTable(i).owner)
    val drained       = channelRows(i) === 0.U && !(io.mem_req(i).read.resp.valid || io.mem_req(i).write.resp.valid ||
      io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid)
    mappingTable(i).valid && ownerFinished && drained
  })

  when(release.asUInt.orR) {
    val channel = PriorityEncoder(release)
    when(mappingTable(channel).isRead) {
      readPortAllocated(mappingTable(channel).id) := false.B
    }.otherwise {
      writePortAllocated(mappingTable(channel).id) := false.B
    }
    mappingTable(channel) := 0.U.asTypeOf(new MappingTableEntry)
  }.elsewhen(requested.asUInt.orR && freePorts.reduce(_ || _)) {
    val channel = PriorityEncoder(freePorts)
    val port    = PriorityEncoder(requested)
    mappingTable(channel).valid  := true.B
    mappingTable(channel).isRead := port < totalRead.U
    mappingTable(channel).id     := Mux(port < totalRead.U, port, port - totalRead.U)
    mappingTable(channel).owner  := Mux(
      port < totalRead.U,
      MuxLookup(port, 0.U)(readPortOwner.zipWithIndex.map { case (ball, i) => i.U -> ball.U }.toSeq),
      MuxLookup(port - totalRead.U, 0.U)(writePortOwner.zipWithIndex.map { case (ball, i) => i.U -> ball.U }.toSeq)
    )
    when(port < totalRead.U) {
      readPortAllocated(port) := true.B
    }.otherwise {
      writePortAllocated(port - totalRead.U) := true.B
    }
  }
}
