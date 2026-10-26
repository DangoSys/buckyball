package framework.core.tbus.router

import framework.lib.arbiter.{Arbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.tbus.configs.{Params => TBusParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain.BankPort
import framework.acc.debug.trace.MTraceDPI
import framework.lib.hash.BankHashMonitor
import framework.acc.memdomain.bank_router.blt.AccPipe
import framework.acc.memdomain.PhysicalBankHash
import framework.acc.memdomain.LayoutConfig
import framework.core.tbus.configs.Layout
import framework.acc.memdomain.bank_router.BankMapping
import framework.ip.memory.mesh_shm.{MeshLocalBankPort, MeshSharedMemIO}
import framework.acc.memdomain.{MvoverISA, MvoverPort}

@instantiable
class Router(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val tbusParam:     TBusParam,
  val endpointIds:   Seq[Int],
  val traceIds:      Seq[Int],
  val diffTest:      Boolean)
    extends Module {
  val externalPhysicalPorts = tbusParam.router.physicalPorts
  require(externalPhysicalPorts >= 0)
  val nThreads              = tbusParam.slots
  val totalBanks            = Layout.totalBank(memParam)
  val totalChannel          = Layout.totalChannel(memParam)

  val sharedMemory = memParam.copy(
    bankEntries = memParam.sharedBankEntries
  )

  val tagBits = math.max(1, log2Ceil(frontendParam.rob_entries))

  val threadIds = VecInit(endpointIds.map(_.U(64.W)))
  require(traceIds.size == endpointIds.size)
  def traceId(owner: UInt): UInt = Mux1H(endpointIds.map(id => owner === id.U), traceIds.map(_.U(64.W)))

  val memoryParams = tbusParam.router.mesh(frontendParam, memParam, tbusParam)

  @public
  val io = IO(new Bundle {

    val memory = Flipped(new MeshSharedMemIO(memoryParams))

    val mem_req            = Vec(totalChannel, Flipped(new BankPort(frontendParam, memParam)))
    val lease              = Option.when(externalPhysicalPorts > 0)(Flipped(new LeasePort))
    val leaseOwnerThreadId = Option.when(externalPhysicalPorts > 0)(Input(UInt(64.W)))

    val physical = Vec(
      externalPhysicalPorts,
      Flipped(new PhysicalPort(memParam.bankWidth))
    )

    val mvover = Flipped(new MvoverPort)

    val localBanks =
      Vec(
        nThreads,
        new MeshLocalBankPort(
          MvoverISA.AddressBits,
          MvoverISA.BankBits,
          memParam.bankWidth,
          tagBits
        )
      )

    val config  = Flipped(Decoupled(new LayoutConfig(frontendParam, memParam)))
    val mapping = Vec(nThreads, Flipped(Decoupled(new BankMapping(frontendParam))))

    val bank_hashes = if (diffTest) Some(Output(Vec(totalBanks, new PhysicalBankHash(frontendParam)))) else None
  })

  require(io.memory.transferCommand.bits.sourceAddr.getWidth <= 16)
  require(io.memory.transferCommand.bits.sourceThread.getWidth <= 8)
  io.memory.transferCommand.valid             := io.mvover.command.valid
  io.memory.transferCommand.bits.sourceThread := io.mvover.command.bits.sourceThread
  io.memory.transferCommand.bits.targetThread := io.mvover.command.bits.targetThread
  io.memory.transferCommand.bits.sourceBank   := io.mvover.command.bits.sourceBank
  io.memory.transferCommand.bits.targetBank   := io.mvover.command.bits.targetBank
  io.memory.transferCommand.bits.sourceAddr   := io.mvover.command.bits.sourceAddr
  io.memory.transferCommand.bits.targetAddr   := io.mvover.command.bits.targetAddr
  io.memory.transferCommand.bits.rows         := io.mvover.command.bits.rows
  io.memory.transferCommand.bits.tag          := 0.U
  io.mvover.command.ready                     := io.memory.transferCommand.ready
  io.mvover.completion.valid                  := io.memory.transferCompletion.valid
  io.mvover.completion.bits                   := io.memory.transferCompletion.bits.error
  io.memory.transferCompletion.ready          := io.mvover.completion.ready
  for ((local, index) <- io.memory.localBanks.zipWithIndex) {
    io.localBanks(index).request.valid  := local.request.valid
    io.localBanks(index).request.bits   := local.request.bits
    local.request.ready                 := io.localBanks(index).request.ready
    local.response.valid                := io.localBanks(index).response.valid
    local.response.bits                 := io.localBanks(index).response.bits
    io.localBanks(index).response.ready := local.response.ready
  }

  val accPipes: Seq[Instance[AccPipe]] = Seq.fill(totalChannel)(Instantiate(new AccPipe(frontendParam, memParam)))

  val hashMonitors: Option[Seq[Instance[BankHashMonitor]]] =
    if (diffTest) {
      Some(Seq.fill(totalBanks)(Instantiate(new BankHashMonitor(sharedMemory))))
    } else {
      None
    }

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces: Seq[Instance[MTraceDPI]] = Seq.fill(totalChannel + externalPhysicalPorts)(Instantiate(new MTraceDPI))
  for (mt <- mtraces) {
    mt.io.clock      := clock
    mt.io.reset      := reset.asBool
    mt.io.is_write   := 0.U
    mt.io.is_shared  := 0.U
    mt.io.channel    := 0.U
    mt.io.thread_id  := 0.U
    mt.io.rob_id     := 0.U
    mt.io.inst_id    := 0.U
    mt.io.vbank_id   := 0.U
    mt.io.pbank_id   := 0.U
    mt.io.group_id   := 0.U
    mt.io.addr       := 0.U
    mt.io.write_mask := 0.U
    mt.io.data_lo    := 0.U
    mt.io.data_hi    := 0.U
    mt.io.enable     := false.B
  }

  // -----------------------------------------------------------------------------
  // Mapping table
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid            = Bool()
    val thread_id        = UInt(64.W)
    val vbank_id         = UInt(frontendParam.bank_id_len.W)
    val architectural_id = UInt(frontendParam.bank_id_len.W)
    val is_multi         = Bool()
    val group_id         = UInt(memParam.groupIdWidth.W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(totalBanks)(0.U.asTypeOf(new MappingTableEntry))))

  val architecturalColumns = RegInit(VecInit(Seq.fill(nThreads)(VecInit(
    Seq.fill(memParam.virtualBankCount)(0.U(
      memParam.groupCountWidth.W
    ))
  ))))

  val committed = RegInit(VecInit(Seq.fill(nThreads)(VecInit(
    (0 until memParam.virtualBankCount).map(_.U(
      frontendParam.bank_id_len.W
    ))
  ))))

  val leases = Option.when(externalPhysicalPorts > 0)(RegInit(VecInit(Seq.fill(totalBanks)(0.U(32.W)))))

  // Record binding and lifetime changes, including aliases with no active accesses.
  for ((entry, physical) <- mappingTable.zipWithIndex) {
    val ownerThread     = PriorityEncoder(VecInit(threadIds.map(_ === entry.thread_id)).asUInt)
    val live            = entry.valid
    val current         = entry.valid && committed(ownerThread)(entry.architectural_id) === entry.vbank_id
    val previous        = RegNext(entry, 0.U.asTypeOf(entry))
    val previousLive    = RegNext(live, false.B)
    val previousCurrent = RegNext(current, false.B)
    val trace           = Instantiate(new MTraceDPI)
    trace.io.clock      := clock
    trace.io.reset      := reset.asBool
    trace.io.is_write   := Mux(entry.valid, 2.U, 3.U)
    trace.io.is_shared  := 1.U
    trace.io.channel    := 0.U
    trace.io.thread_id  := traceId(entry.thread_id)
    trace.io.rob_id     := 0.U
    trace.io.inst_id    := 0.U
    trace.io.vbank_id   := entry.vbank_id
    trace.io.pbank_id   := physical.U
    trace.io.group_id   := entry.group_id
    trace.io.addr       := entry.architectural_id
    trace.io.write_mask := 0.U
    trace.io.data_lo    := live.asUInt
    trace.io.data_hi    := current.asUInt
    trace.io.enable     := (entry.valid || previous.valid) &&
      (entry.asUInt =/= previous.asUInt || live =/= previousLive || current =/= previousCurrent)
  }

  io.lease.foreach { port =>
    val pending      = RegInit(false.B)
    val result       = Reg(UInt(64.W))
    val ownerThread  = PriorityEncoder(VecInit(threadIds.map(_ === io.leaseOwnerThreadId.get)).asUInt)
    val currentAlias = committed(ownerThread)(port.request.bits.vbank)
    val matches      = VecInit(mappingTable.zipWithIndex.map { case (entry, physical) =>
      entry.valid &&
        entry.thread_id === io.leaseOwnerThreadId.get && entry.architectural_id === port.request.bits.vbank &&
        entry.group_id === port.request.bits.group &&
        Mux(port.request.bits.release, leases.get(physical) =/= 0.U, entry.vbank_id === currentAlias)
    })
    val index        = PriorityEncoder(matches)
    port.request.ready               := !pending && !io.config.valid && !reset.asBool
    port.response.valid              := pending && !reset.asBool
    port.response.bits               := result
    when(port.request.fire) {
      assert(PopCount(matches) === 1.U, "Shared lease requires one allocated owner/bank/group")
      val count = leases.get(index)
      when(port.request.bits.release) {
        assert(count =/= 0.U, "Shared lease release has no export")
        count  := count - 1.U
        result := 0.U
      }.otherwise {
        assert(count =/= "hffffffff".U, "Shared lease count overflow")
        count  := count + 1.U
        result := index * (memParam.sharedBankEntries * (memParam.bankWidth / 8)).U(
          64.W
        )
      }
      pending := true.B
    }
    when(port.response.fire)(pending := false.B)
  }

  val rowBits     = log2Ceil(memParam.sharedBankEntries)
  val bytesPerRow = memParam.bankWidth / 8
  val byteBits    = log2Ceil(bytesPerRow)
  for (i <- 0 until externalPhysicalPorts) {
    val physical  = io.physical(i)
    val channel   = io.memory.external(i)
    val bank      = physical.request.bits.address >> (rowBits + byteBits)
    val legal     = bank < totalBanks.U && physical.request.bits.address(byteBits - 1, 0) === 0.U
    val bankIndex = bank(math.max(1, log2Ceil(totalBanks)) - 1, 0)
    val allocated = VecInit(mappingTable.map(_.valid))(bankIndex)
    when(physical.request.valid && !reset.asBool) {
      assert(legal && allocated, "C2C physical access requires an allocated shared bank and aligned address")
    }
    channel.request.valid        := physical.request.valid && legal && allocated
    channel.request.bits         := 0.U.asTypeOf(channel.request.bits)
    channel.request.bits.bank    := bankIndex
    channel.request.bits.tuser   := Cat(
      physical.request.bits.address(rowBits + byteBits - 1, byteBits).pad(16),
      physical.request.bits.write
    )
    channel.request.bits.data    := physical.request.bits.data
    channel.request.bits.mask    := physical.request.bits.mask
    channel.request.bits.tag     := 0.U
    channel.request.bits.tlast   := true.B
    physical.request.ready       := channel.request.ready && legal && allocated
    physical.response.valid      := channel.response.valid
    physical.response.bits.data  := channel.response.bits.data
    physical.response.bits.error := channel.response.bits.error
    channel.response.ready       := physical.response.ready
  }

  for (i <- 0 until externalPhysicalPorts) {
    val physical = io.physical(i)
    val held     = Reg(new PhysicalRequest(memParam.bankWidth))
    val entry    = Reg(new MappingTableEntry)
    val pending  = RegInit(false.B)
    val bankBits = log2Ceil(
      memParam.sharedBankEntries * (memParam.bankWidth / 8)
    )
    when(physical.request.fire) {
      assert(!pending, "Physical shared port accepts one outstanding request")
      held    := physical.request.bits
      entry   := mappingTable((physical.request.bits.address >> bankBits)(math.max(1, log2Ceil(totalBanks)) - 1, 0))
      pending := true.B
    }
    when(physical.response.valid && !reset.asBool)(assert(pending))
    when(physical.response.fire)(pending := false.B)
    val trace    = mtraces(totalChannel + i)
    val value    = Mux(held.write, held.data, physical.response.bits.data)
    trace.io.is_write   := held.write.asUInt
    trace.io.is_shared  := 1.U
    trace.io.channel    := (totalChannel + i).U
    trace.io.thread_id  := traceId(entry.thread_id)
    trace.io.vbank_id   := entry.vbank_id
    trace.io.pbank_id   := held.address >> bankBits
    trace.io.group_id   := entry.group_id
    trace.io.addr       := (held.address >> 4) & (memParam.sharedBankEntries - 1).U
    trace.io.write_mask := Mux(held.write, held.mask, 0.U)
    trace.io.data_lo    := value(63, 0)
    trace.io.data_hi    := value(127, 64)
    trace.io.enable     := physical.response.fire && !physical.response.bits.error
  }

  def addEntry(
    thread_id: UInt,
    vbank_id:  UInt,
    pbank_id:  UInt,
    is_multi:  Bool,
    group_id:  UInt
  ): Unit = {
    val duplicate = mappingTable.map(entry =>
      entry.valid &&
        !(group_id === 0.U && entry.thread_id === thread_id && entry.vbank_id === vbank_id) &&
        (entry.thread_id === thread_id) &&
        (entry.vbank_id === vbank_id) &&
        (entry.group_id === group_id)
    ).reduce(_ || _)
    when(duplicate) {
      assert(false.B, "Router duplicate allocation: thread=%d vbank=%d group=%d\n", thread_id, vbank_id, group_id)
    }

    val entry = mappingTable(pbank_id)
    leases.foreach(counts => assert(counts(pbank_id) === 0.U, "Shared bank allocation replaces live exports"))
    entry.valid            := true.B
    entry.thread_id        := thread_id
    entry.vbank_id         := vbank_id
    entry.architectural_id := vbank_id
    entry.is_multi         := is_multi
    entry.group_id         := group_id
  }

  def deleteEntry(thread_id: UInt, vbank_id: UInt): Unit = {
    val thread = PriorityEncoder(VecInit(threadIds.map(_ === thread_id)).asUInt)
    val found  = architecturalColumns(thread)(vbank_id) =/= 0.U
    when(!found) {
      assert(false.B, "Router release missing allocation: thread=%d vbank=%d\n", thread_id, vbank_id)
    }
    clearVbank(thread_id, vbank_id)
  }

  def clearVbank(thread_id: UInt, vbank_id: UInt): Unit = {
    for (i <- 0 until totalBanks) {
      when(mappingTable(i).valid && mappingTable(i).architectural_id === vbank_id && mappingTable(
        i
      ).thread_id === thread_id) {
        leases.foreach(counts => assert(counts(i) === 0.U, "Shared bank mapping changed with live exports"))
        mappingTable(i).valid := false.B
      }
    }
  }

  // -----------------------------------------------------------------------------
  // Default Value
  // -----------------------------------------------------------------------------

  for (i <- 0 until totalChannel) {
    accPipes(i).io.mem_req.write <> io.mem_req(i).write
    accPipes(i).io.mem_req.read <> io.mem_req(i).read
    accPipes(i).io.mem_req.bank_id   := io.mem_req(i).bank_id
    accPipes(i).io.mem_req.group_id  := io.mem_req(i).group_id
    accPipes(i).io.mem_req.thread_id := io.mem_req(i).thread_id

    // Bank-side defaults (only driven when a bank is actually connected)
    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare
  }

  val realloc = io.config.bits.group_id === 0.U

  val freeBanks = VecInit(mappingTable.map(entry => !entry.valid)).asUInt

  val freePbankMask = VecInit(mappingTable.zipWithIndex.map { case (entry, physical) =>
    freeBanks(physical) ||
      (realloc && entry.thread_id === io.config.bits.thread_id && entry.architectural_id === io.config.bits.vbank_id)
  })

  val hasFreePbank = freePbankMask.asUInt.orR
  io.config.ready := !io.config.bits.alloc || hasFreePbank
  when(io.config.valid && io.config.bits.alloc && !hasFreePbank) {
    assert(false.B, "Router allocation failed: no free physical shared bank\n")
  }

  // -----------------------------------------------------------------------------
  // Bank Alloc/Release
  // -----------------------------------------------------------------------------

  hashMonitors.foreach(_.foreach(_.io.bind := false.B))

  val mappings = Instantiate(new Arbiter(new BankMapping(frontendParam), nThreads))
  for (thread <- 0 until nThreads) { mappings.io.in(thread) <> io.mapping(thread) }
  val change  = mappings.io.out.bits
  val owner   = mappings.io.chosen
  val columns = architecturalColumns(owner)(change.bank)

  val leaseBusy = VecInit(mappingTable.zipWithIndex.map { case (entry, physical) =>
    val held = if (externalPhysicalPorts > 0) leases.get(physical) =/= 0.U else false.B
    entry.valid && entry.thread_id === change.threadId && entry.vbank_id === change.previous && held
  }).asUInt.orR

  val reusableBanks = VecInit(mappingTable.map(entry =>
    !entry.valid || (change.reuse && entry.thread_id === change.threadId && entry.vbank_id === change.previous)
  )).asUInt

  mappings.io.out.ready := !io.config.valid && (!change.allocate || PopCount(reusableBanks) >= columns) &&
    (!(change.retire || change.reuse) || !leaseBusy)
  when(mappings.io.out.valid && change.allocate) {
    assert(columns =/= 0.U && columns <= totalBanks.U, "Renamed shared bank requires an allocated layout")
  }
  when(mappings.io.out.fire && change.retire) {
    committed(owner)(change.bank) := change.vbank
    for (entry <- mappingTable) {
      when(entry.valid && entry.thread_id === change.threadId && entry.vbank_id === change.previous) {
        entry.valid := false.B
      }
    }
  }
  val remaining = Wire(Vec(totalBanks + 1, UInt(totalBanks.W)))
  remaining(0)          := reusableBanks
  for (group <- 0 until totalBanks) {
    val physical = PriorityEncoder(remaining(group))
    remaining(group + 1) := remaining(group) & ~UIntToOH(physical, totalBanks)
    when(mappings.io.out.fire && change.allocate && group.U < columns) {
      assert(change.allocate && columns =/= 0.U, "Shared alias requires an allocated layout")
      addEntry(change.threadId, change.vbank, physical, columns > 1.U, group.U)
      mappingTable(physical).architectural_id := change.bank
      hashMonitors.foreach { hashes =>
        for (bank <- 0 until totalBanks) { when(physical === bank.U)(hashes(bank).io.bind := true.B) }
      }
    }
  }

  when(io.config.fire) {
    val thread = PriorityEncoder(VecInit(threadIds.map(_ === io.config.bits.thread_id)).asUInt)
    when(io.config.bits.transfer) {
      val source      = io.config.bits.source_bank_id
      val target      = io.config.bits.vbank_id
      val owner       = io.config.bits.thread_id
      val sourceCount = architecturalColumns(thread)(source)
      val targetCount = architecturalColumns(thread)(target)
      val totalCount  = sourceCount +& targetCount
      architecturalColumns(thread)(source) := 0.U
      architecturalColumns(thread)(target) := totalCount
      assert(source =/= target && sourceCount =/= 0.U, "Shared bank transfer requires a distinct allocated source")
      assert(totalCount <= totalBanks.U, "Shared bank transfer exceeds physical bank count")
      for ((entry, physical) <- mappingTable.zipWithIndex) {
        when(entry.valid && entry.thread_id === owner && entry.architectural_id === source) {
          leases.foreach(counts => assert(counts(physical) === 0.U, "Shared bank transfer source has live exports"))
          entry.vbank_id         := target
          entry.architectural_id := target
          entry.group_id         := targetCount + entry.group_id
          entry.is_multi         := totalCount > 1.U
        }
        when(entry.valid && entry.thread_id === owner && entry.architectural_id === target) {
          entry.vbank_id := target
          entry.is_multi := totalCount > 1.U
        }
      }
    }.elsewhen(io.config.bits.alloc) {
      architecturalColumns(thread)(io.config.bits.vbank_id) := io.config.bits.group_id +& 1.U
      when(io.config.bits.group_id === 0.U) {
        clearVbank(io.config.bits.thread_id, io.config.bits.vbank_id)
      }
      val pbankId = PriorityEncoder(freePbankMask)
      hashMonitors.foreach { hashes =>
        for (i <- 0 until totalBanks) {
          hashes(i).io.bind := pbankId === i.U
        }
      }
      printf(
        p"[Router][ALLOC] thread=${io.config.bits.thread_id} vbank=0x${Hexadecimal(io.config.bits.vbank_id)} " +
          p"group=${io.config.bits.group_id} pbank=$pbankId is_multi=${io.config.bits.is_multi}\n"
      )
      addEntry(
        io.config.bits.thread_id,
        io.config.bits.vbank_id,
        pbankId,
        io.config.bits.is_multi,
        io.config.bits.group_id
      )
    }.otherwise {
      architecturalColumns(thread)(io.config.bits.vbank_id) := 0.U
      printf(
        p"[Router][RELEASE] thread=${io.config.bits.thread_id} vbank=0x${Hexadecimal(io.config.bits.vbank_id)}\n"
      )
      deleteEntry(io.config.bits.thread_id, io.config.bits.vbank_id)
    }
  }

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  def emitTrace(
    ch:        Int,
    isWrite:   UInt,
    pbankId:   UInt,
    addr:      UInt,
    writeMask: UInt,
    dataLo:    UInt,
    dataHi:    UInt,
    en:        Bool
  ): Unit = {
    mtraces(ch).io.is_write   := isWrite
    mtraces(ch).io.is_shared  := io.mem_req(ch).is_shared.asUInt
    mtraces(ch).io.channel    := ch.U
    mtraces(ch).io.thread_id  := traceId(io.mem_req(ch).thread_id)
    mtraces(ch).io.rob_id     := io.mem_req(ch).rob_id
    mtraces(ch).io.inst_id    := io.mem_req(ch).inst_id
    mtraces(ch).io.vbank_id   := io.mem_req(ch).bank_id
    mtraces(ch).io.pbank_id   := pbankId
    mtraces(ch).io.group_id   := io.mem_req(ch).group_id
    mtraces(ch).io.addr       := addr
    mtraces(ch).io.write_mask := writeMask
    mtraces(ch).io.data_lo    := dataLo
    mtraces(ch).io.data_hi    := dataHi
    mtraces(ch).io.enable     := en
  }

  for (i <- 0 until totalChannel) {
    val activeHart  = Mux(accPipes(i).io.busy, accPipes(i).io.thread_id, io.mem_req(i).thread_id)
    val activeBank  = Mux(accPipes(i).io.busy, accPipes(i).io.bank_id, io.mem_req(i).bank_id)
    val activeGroup = Mux(accPipes(i).io.busy, accPipes(i).io.group_id, io.mem_req(i).group_id)
    val req_valid   = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid || accPipes(i).io.busy

    val tracePbankId = Wire(UInt(32.W))
    tracePbankId := 0.U
    for (j <- 0 until totalBanks) {
      val trace_hit_bank = mappingTable(j).valid &&
        (mappingTable(j).thread_id === activeHart) &&
        (mappingTable(j).vbank_id === activeBank) &&
        (!mappingTable(j).is_multi ||
          (mappingTable(j).is_multi && (mappingTable(j).group_id === activeGroup)))
      when(trace_hit_bank) {
        tracePbankId := j.U
      }
    }

    // Memory trace: read request
    when(io.mem_req(i).read.req.fire) {
      emitTrace(i, 0.U, tracePbankId, io.mem_req(i).read.req.bits.addr, 0.U, 0.U, 0.U, true.B)
    }

    // Arrival trace: observe acceptance at the selected shared SPM SRAM port.
    when(accPipes(i).io.sramWrite.req.fire) {
      emitTrace(
        i,
        1.U,
        tracePbankId,
        accPipes(i).io.sramWrite.req.bits.addr,
        accPipes(i).io.sramWrite.req.bits.mask.asUInt,
        accPipes(i).io.sramWrite.req.bits.data(63, 0),
        accPipes(i).io.sramWrite.req.bits.data(127, 64),
        true.B
      )
    }

    val channel  = io.memory.channels(i)
    val mapped   = mappingTable.map(entry =>
      entry.valid &&
        entry.thread_id === activeHart &&
        entry.vbank_id === activeBank &&
        (!entry.is_multi || entry.group_id === activeGroup)
    ).reduce(_ || _)
    val writeReq = accPipes(i).io.sramWrite.req.valid
    val nextTag  = RegInit(0.U(tagBits.W))

    channel.request.valid                  := mapped && (writeReq || accPipes(i).io.sramRead.req.valid)
    channel.request.bits.bank              := tracePbankId
    channel.request.bits.tuser             := Cat(
      Mux(writeReq, accPipes(i).io.sramWrite.req.bits.addr, accPipes(i).io.sramRead.req.bits.addr),
      writeReq
    )
    channel.request.bits.tlast             := true.B
    channel.request.bits.data              := accPipes(i).io.sramWrite.req.bits.data
    channel.request.bits.mask              := accPipes(i).io.sramWrite.req.bits.mask.asUInt
    channel.request.bits.tag               := nextTag
    accPipes(i).io.sramRead.req.ready      := mapped && !writeReq && channel.request.ready
    accPipes(i).io.sramWrite.req.ready     := mapped && channel.request.ready
    accPipes(i).io.sramRead.resp.valid     := channel.response.valid && !channel.response.bits.write
    accPipes(i).io.sramRead.resp.bits.data := channel.response.bits.data
    accPipes(i).io.sramWrite.resp.valid    := channel.response.valid && channel.response.bits.write
    accPipes(i).io.sramWrite.resp.bits.ok  := !channel.response.bits.error
    channel.response.ready                 := Mux(
      channel.response.bits.write,
      accPipes(i).io.sramWrite.resp.ready,
      accPipes(i).io.sramRead.resp.ready
    )
    when(channel.request.fire) {
      nextTag := nextTag + 1.U
    }

  }

  io.memory.bankWrites.foreach(_.ready := true.B)

  hashMonitors.foreach { hashes =>
    for (j <- 0 until totalBanks) {
      val monitor = hashes(j)

      val write = io.memory.bankWrites(j)
      write.ready                := monitor.io.write.ready
      monitor.io.write.valid     := write.valid
      monitor.io.write.bits.addr := write.bits.addr
      monitor.io.write.bits.mask := write.bits.mask.asBools
      monitor.io.write.bits.data := write.bits.data

    }
  }

  io.bank_hashes.foreach { states =>
    for (i <- 0 until totalBanks) {
      states(i).valid      := mappingTable(i).valid
      states(i).threadId   := mappingTable(i).thread_id
      states(i).vbankId    := mappingTable(i).vbank_id
      states(i).pbankId    := i.U
      states(i).groupId    := mappingTable(i).group_id
      states(i).statusHash := hashMonitors.get(i).io.statusHash
    }
  }
}

/** Core shared SRAM byte space; not a Thread virtual-bank or private-bank address. */
class PhysicalRequest(dataBits: Int) extends Bundle {

  val address = UInt(64.W)
  val write   = Bool()
  val data    = UInt(dataBits.W)
  val mask    = UInt((dataBits / 8).W)
}

class PhysicalResponse(dataBits: Int) extends Bundle {

  val data  = UInt(dataBits.W)
  val error = Bool()
}

/** One outstanding transaction per port; a write response acknowledges actual SRAM completion. */
class PhysicalPort(dataBits: Int) extends Bundle {

  val request  = Decoupled(new PhysicalRequest(dataBits))
  val response = Flipped(Decoupled(new PhysicalResponse(dataBits)))
}

class LeaseRequest extends Bundle {
  val endpoint = UInt(32.W)
  val vbank    = UInt(16.W)
  val group    = UInt(16.W)
  val release  = Bool()
}

/** One outstanding mapping lease operation; export returns its physical byte base. */
class LeasePort extends Bundle {
  val request  = Decoupled(new LeaseRequest)
  val response = Flipped(Decoupled(UInt(64.W)))
}
