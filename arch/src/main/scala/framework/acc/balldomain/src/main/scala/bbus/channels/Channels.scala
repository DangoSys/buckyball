package framework.acc.balldomain.bbus.channels

import chisel3.experimental.hierarchy.Instantiate

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.lib.queue.Queue
import chisel3._
import chisel3.util._
import framework.acc.memdomain.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.memdomain.BankPort

/**
 * Channel allocation and release
 * Connects memory clients to the layout table
 *
 * Configured Ball read and write ports retain their channels until completion.
 * The port ordering is determined by the Ball connection configuration.
 * Requests retain their channel until the owning Ball completes and responses drain.
 */
@instantiable
class Channels(fp: FrontendParam, mp: MemDomainParam, bp: BallDomainParam) extends Module {
  // Each configured Ball owns its declared read and write ports.
  val ownerCount = bp.balls.size
  val totalRead  = bp.balls.map(_.inBW).sum
  val totalWrite = bp.balls.map(_.outBW).sum

  @public
  val io = IO(new Bundle {
    // Ball read and write ports in configuration order.
    val bankRead  = Vec(totalRead, new BankRead(fp, mp))
    val bankWrite = Vec(totalWrite, new BankWrite(fp, mp))
    val active    = Input(Vec(bp.balls.size, Bool()))

    val instructionIds = Input(Vec(ownerCount, UInt(64.W)))

    // Output to the layout table
    val mem_req = Vec(mp.bankRequestPorts, new BankPort(fp, mp))
  })

  val readPortOwner  = bp.balls.zipWithIndex.flatMap { case (ball, owner) => Seq.fill(ball.inBW)(owner) }
  val writePortOwner = bp.balls.zipWithIndex.flatMap { case (ball, owner) => Seq.fill(ball.outBW)(owner) }

  val readInstructionIds = Wire(Vec(totalRead, UInt(64.W)))
  val readClients        = Wire(Vec(totalRead, new BankRead(fp, mp)))
  for (i <- 0 until totalRead) {
    val source  = io.bankRead(i)
    val request = Wire(Decoupled(new Bundle {
      val bank    = chiselTypeOf(source.bank_id)
      val rob     = chiselTypeOf(source.rob_id)
      val group   = chiselTypeOf(source.group_id)
      val address = chiselTypeOf(source.io.req.bits)
      val instId  = UInt(64.W)
    }))
    request.valid := source.io.req.valid
    request.bits.bank    := source.bank_id
    request.bits.rob     := source.rob_id
    request.bits.group   := source.group_id
    request.bits.address := source.io.req.bits
    request.bits.instId  := io.instructionIds(readPortOwner(i))
    source.io.req.ready  := request.ready
    val queue = Instantiate(new Queue(chiselTypeOf(request.bits), 8, false, false))
    queue.io.enq <> request
    readInstructionIds(i) := queue.io.deq.bits.instId
    val target = readClients(i)
    target.io.req.valid := queue.io.deq.valid
    target.io.req.bits  := queue.io.deq.bits.address
    target.bank_id      := queue.io.deq.bits.bank
    target.rob_id       := queue.io.deq.bits.rob
    target.group_id     := queue.io.deq.bits.group
    queue.io.deq.ready  := target.io.req.ready
    source.io.resp <> target.io.resp
  }

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
    Seq.fill(mp.bankRequestPorts)(0.U.asTypeOf(new MappingTableEntry))
  ))

  // A route stays assigned until its Ball releases it, so maintain the reverse ownership at allocation time. This keeps
  // readiness and pending detection from repeatedly scanning every backend
  // request port for the same port.
  val portSlots          = math.max(totalRead, totalWrite)
  val readPortAllocated  = RegInit(VecInit(Seq.fill(portSlots)(false.B)))
  val writePortAllocated = RegInit(VecInit(Seq.fill(portSlots)(false.B)))
  val channelPending     =
    RegInit(VecInit(Seq.fill(mp.bankRequestPorts)(false.B)))

  // Allocate or release one channel per cycle, retaining its owner through response drain.
  for (i <- 0 until totalRead) {
    readClients(i).io.req.ready  := false.B
    readClients(i).io.resp.valid := false.B
    readClients(i).io.resp.bits  := DontCare

  }

  for (i <- 0 until totalWrite) {
    io.bankWrite(i).io.req.ready  := false.B
    io.bankWrite(i).io.resp.valid := false.B
    io.bankWrite(i).io.resp.bits  := DontCare
  }

  val pendingReads  = VecInit((0 until totalRead).map(i => io.active(readPortOwner(i)) && !readPortAllocated(i)))
  val pendingWrites = VecInit((0 until totalWrite).map(i => io.active(writePortOwner(i)) && !writePortAllocated(i)))

  val pending   = VecInit(pendingReads ++ pendingWrites)
  val freePorts = mappingTable.map(entry => !entry.valid)

  // Connect mapped entries to layout request ports
  for (i <- 0 until mp.bankRequestPorts) {
    io.mem_req(i).read.req.valid   := false.B
    io.mem_req(i).read.req.bits    := DontCare
    io.mem_req(i).read.resp.ready  := false.B
    io.mem_req(i).write.req.valid  := false.B
    io.mem_req(i).write.req.bits   := DontCare
    io.mem_req(i).write.resp.ready := false.B
    io.mem_req(i).bank_id          := 0.U
    io.mem_req(i).group_id         := 0.U
    io.mem_req(i).is_shared        := false.B
    io.mem_req(i).thread_id        := fp.ownerId.U(64.W)
    io.mem_req(i).rob_id           := 0.U
    io.mem_req(i).inst_id          := 0.U

    val isRead    = mappingTable(i).isRead
    val rid       = mappingTable(i).id
    val wid       = mappingTable(i).id
    val ballRead  = readClients(rid).io
    val ballWrite = io.bankWrite(wid).io
    val rbank_id  = readClients(rid).bank_id
    val wbank_id  = io.bankWrite(wid).bank_id
    val rgroup_id = readClients(rid).group_id
    val wgroup_id = io.bankWrite(wid).group_id
    val rrob_id   = readClients(rid).rob_id
    val wrob_id   = io.bankWrite(wid).rob_id

    when(mappingTable(i).valid) {
      when(isRead) {
        io.mem_req(i).read <> ballRead
        io.mem_req(i).bank_id  := rbank_id
        io.mem_req(i).group_id := rgroup_id
        io.mem_req(i).rob_id   := rrob_id
        io.mem_req(i).inst_id  := readInstructionIds(rid)
      }.otherwise {
        io.mem_req(i).write <> ballWrite
        io.mem_req(i).bank_id  := wbank_id
        io.mem_req(i).group_id := wgroup_id
        io.mem_req(i).rob_id   := wrob_id
        io.mem_req(i).inst_id  := io.instructionIds(mappingTable(i).owner)
      }
    }
  }

  for (i <- 0 until mp.bankRequestPorts) {
    when(io.mem_req(i).read.req.fire || io.mem_req(i).write.req.fire)(channelPending(i)   := true.B)
    when(io.mem_req(i).read.resp.fire || io.mem_req(i).write.resp.fire)(channelPending(i) := false.B)
  }

  val release = VecInit((0 until mp.bankRequestPorts).map { i =>
    val ownerFinished = !io.active(mappingTable(i).owner)
    val drained       = !channelPending(i) && !(io.mem_req(i).read.resp.valid || io.mem_req(i).write.resp.valid ||
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
  }.elsewhen(pending.asUInt.orR && freePorts.reduce(_ || _)) {
    val channel = PriorityEncoder(freePorts)
    val port    = PriorityEncoder(pending)
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
