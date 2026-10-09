package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.{MTraceDPI, MemRequestIO}
import sims.hash.BankHashMonitor
import framework.memdomain.backend.accpipe.AccPipe
import framework.memdomain.backend.banks.SramBank
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.frontend.mem.MemConfigerIO
import framework.top.GlobalConfig
import memcore.memory.mesh_shm.{MeshCoreAttachment, MeshLocalBankPort, MeshSharedMem, MeshSharedMemParams}
import framework.memdomain.isa.{MvoverISA, MvoverPort}

@instantiable
class SharedMemBackend(val b: GlobalConfig, useMesh: Boolean = false, externalPhysicalPorts: Int = 0) extends Module {
  require(externalPhysicalPorts >= 0 && (externalPhysicalPorts == 0 || useMesh))
  val nCores       = b.memDomain.nCores
  val totalBanks   = SharedMemLayout.totalBank(b)
  val totalChannel = SharedMemLayout.totalChannel(b)
  val sharedConfig = b.copy(memDomain = b.memDomain.copy(bankEntries = b.memDomain.sharedBankEntries))
  val tagBits      = math.max(1, log2Ceil(b.frontend.rob_entries))

  @public
  val io = IO(new Bundle {
    val mem_req             = Vec(totalChannel, Flipped(new MemRequestIO(b)))
    val lease               = Option.when(externalPhysicalPorts > 0)(Flipped(new SharedLeasePort))
    val leaseOwnerHartId    = Option.when(externalPhysicalPorts > 0)(Input(UInt(b.tile.xLen.W)))
    val physical            = Vec(externalPhysicalPorts, Flipped(new SharedPhysicalPort(b.memDomain.bankWidth)))
    val physicalOwnerHartId = if (externalPhysicalPorts > 0) Some(Input(UInt(b.tile.xLen.W))) else None
    val mvover              = Flipped(new MvoverPort)
    val localBanks          =
      Vec(nCores, new MeshLocalBankPort(MvoverISA.AddressBits, MvoverISA.BankBits, b.memDomain.bankWidth, tagBits))
    val config              = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_valid       = Input(Vec(nCores, Bool()))
    val query_hart_id     = Input(Vec(nCores, UInt(b.tile.xLen.W)))
    val query_vbank_id    = Input(Vec(nCores, UInt(b.memDomain.vbankIdWidth.W)))
    val query_group_count = Output(Vec(nCores, UInt(b.memDomain.groupCountWidth.W)))
    val bank_hashes       = if (b.sim.diffTest) Some(Output(Vec(totalBanks, new PhysicalBankHash(b)))) else None
  })

  val meshParams =
    if (useMesh) {
      require(b.memDomain.bankWidth == 128 && b.memDomain.bankMaskLen == 16)
      require(b.memDomain.sharedInputChannels % b.memDomain.computeCoreIds.size == 0)
      val physicalBanks = totalBanks
      val rows          = math.ceil(math.sqrt(physicalBanks.toDouble)).toInt
      val columns       = (physicalBanks + rows - 1) / rows
      val perCore       = SharedMemLayout.channelPerHart(b)
      Some(MeshSharedMemParams(
        rows = rows,
        cols = columns,
        entriesPerBank = b.memDomain.sharedBankEntries,
        dataBits = b.memDomain.bankWidth,
        tagBits = tagBits,
        localBankBits = MvoverISA.BankBits,
        cores = (0 until nCores).map { core =>
          val computeIndex = b.memDomain.computeCoreIds.indexOf(core)
          MeshCoreAttachment(if (computeIndex < 0) Seq.empty
          else (0 until perCore).map(ch => (computeIndex * perCore + ch) % totalBanks))
        },
        externalChannels = externalPhysicalPorts,
        visibleBanks = totalBanks
      ))
    } else None

  val mesh: Option[Instance[MeshSharedMem]] = meshParams.map(p => Instantiate(new MeshSharedMem(p)))
  mesh.foreach { network =>
    require(network.io.transferCommand.bits.sourceAddr.getWidth <= 16)
    require(network.io.transferCommand.bits.sourceCore.getWidth <= 8)
    network.io.transferCommand.valid           := io.mvover.command.valid
    network.io.transferCommand.bits.sourceCore := io.mvover.command.bits.sourceCore
    network.io.transferCommand.bits.targetCore := io.mvover.command.bits.targetCore
    network.io.transferCommand.bits.sourceBank := io.mvover.command.bits.sourceBank
    network.io.transferCommand.bits.targetBank := io.mvover.command.bits.targetBank
    network.io.transferCommand.bits.sourceAddr := io.mvover.command.bits.sourceAddr
    network.io.transferCommand.bits.targetAddr := io.mvover.command.bits.targetAddr
    network.io.transferCommand.bits.rows       := io.mvover.command.bits.rows
    network.io.transferCommand.bits.tag        := 0.U
    io.mvover.command.ready                    := network.io.transferCommand.ready
    io.mvover.completion.valid                 := network.io.transferCompletion.valid
    io.mvover.completion.bits                  := network.io.transferCompletion.bits.error
    network.io.transferCompletion.ready        := io.mvover.completion.ready
    for ((local, index) <- network.io.localBanks.zipWithIndex) {
      io.localBanks(index).request.valid  := local.request.valid
      io.localBanks(index).request.bits   := local.request.bits
      local.request.ready                 := io.localBanks(index).request.ready
      local.response.valid                := io.localBanks(index).response.valid
      local.response.bits                 := io.localBanks(index).response.bits
      io.localBanks(index).response.ready := local.response.ready
    }
  }
  if (!useMesh) {
    io.mvover.command.ready    := false.B
    io.mvover.completion.valid := false.B
    io.mvover.completion.bits  := false.B
    for (local <- io.localBanks) {
      local.request.valid  := false.B
      local.request.bits   := 0.U.asTypeOf(local.request.bits)
      local.response.ready := false.B
    }
  }

  val banks:    Seq[Instance[SramBank]] =
    if (useMesh) Seq.empty else Seq.fill(totalBanks)(Instantiate(new SramBank(sharedConfig)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(totalChannel)(Instantiate(new AccPipe(b)))

  val hashMonitors: Option[Seq[Instance[BankHashMonitor]]] =
    if (b.sim.diffTest) {
      Some(Seq.fill(totalBanks)(Instantiate(new BankHashMonitor(sharedConfig))))
    } else {
      None
    }

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces = Seq.fill(totalChannel + externalPhysicalPorts)(Module(new MTraceDPI))
  for (mt <- mtraces) {
    mt.io.clock      := clock
    mt.io.reset      := reset.asBool
    mt.io.is_write   := 0.U
    mt.io.is_shared  := 0.U
    mt.io.channel    := 0.U
    mt.io.hart_id    := 0.U
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
    val valid    = Bool()
    val hart_id  = UInt(b.tile.xLen.W)
    val vbank_id = UInt(b.memDomain.vbankIdWidth.W)
    val is_multi = Bool()
    val group_id = UInt(b.memDomain.groupIdWidth.W)
  }

  val mappingTable = RegInit(VecInit(Seq.fill(totalBanks)(0.U.asTypeOf(new MappingTableEntry))))
  val leases       = Option.when(externalPhysicalPorts > 0)(RegInit(VecInit(Seq.fill(totalBanks)(0.U(32.W)))))
  io.lease.foreach { port =>
    val pending = RegInit(false.B)
    val result  = Reg(UInt(64.W))
    val matches = VecInit(mappingTable.map(entry =>
      entry.valid &&
        entry.hart_id === io.leaseOwnerHartId.get && entry.vbank_id === port.request.bits.vbank &&
        entry.group_id === port.request.bits.group
    ))
    val index   = PriorityEncoder(matches)
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
        result := index * (b.memDomain.sharedBankEntries * (b.memDomain.bankWidth / 8)).U(64.W)
      }
      pending := true.B
    }
    when(port.response.fire)(pending := false.B)
  }

  mesh.foreach { network =>
    val rowBits     = log2Ceil(b.memDomain.sharedBankEntries)
    val bytesPerRow = b.memDomain.bankWidth / 8
    val byteBits    = log2Ceil(bytesPerRow)
    for (i <- 0 until externalPhysicalPorts) {
      val physical  = io.physical(i)
      val channel   = network.io.external(i)
      val bank      = physical.request.bits.address >> (rowBits + byteBits)
      val legal     = bank < totalBanks.U && physical.request.bits.address(byteBits - 1, 0) === 0.U
      val bankIndex = bank(math.max(1, log2Ceil(totalBanks)) - 1, 0)
      val allocated = VecInit(mappingTable.map(_.valid))(bankIndex)
      when(physical.request.valid && !reset.asBool) {
        assert(legal && allocated, "T2T physical access requires an allocated shared bank and aligned address")
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
  }
  for (i <- 0 until externalPhysicalPorts) {
    val physical = io.physical(i)
    val held     = Reg(new SharedPhysicalRequest(b.memDomain.bankWidth))
    val entry    = Reg(new MappingTableEntry)
    val pending  = RegInit(false.B)
    val bankBits = log2Ceil(b.memDomain.sharedBankEntries * (b.memDomain.bankWidth / 8))
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
    trace.io.hart_id    := io.physicalOwnerHartId.get
    trace.io.vbank_id   := entry.vbank_id
    trace.io.pbank_id   := held.address >> bankBits
    trace.io.group_id   := entry.group_id
    trace.io.addr       := (held.address >> 4) & (b.memDomain.sharedBankEntries - 1).U
    trace.io.write_mask := Mux(held.write, held.mask, 0.U)
    trace.io.data_lo    := value(63, 0)
    trace.io.data_hi    := value(127, 64)
    trace.io.enable     := physical.response.fire && !physical.response.bits.error
  }

  def addEntry(
    hart_id:  UInt,
    vbank_id: UInt,
    pbank_id: UInt,
    is_multi: Bool,
    group_id: UInt
  ): Unit = {
    val duplicate = mappingTable.map(entry =>
      entry.valid &&
        !(group_id === 0.U && entry.hart_id === hart_id && entry.vbank_id === vbank_id) &&
        (entry.hart_id === hart_id) &&
        (entry.vbank_id === vbank_id) &&
        (entry.group_id === group_id)
    ).reduce(_ || _)
    when(duplicate) {
      assert(false.B, "SharedMemBackend duplicate allocation: hart=%d vbank=%d group=%d\n", hart_id, vbank_id, group_id)
    }

    val entry = mappingTable(pbank_id)
    leases.foreach(counts => assert(counts(pbank_id) === 0.U, "Shared bank allocation replaces live exports"))
    entry.valid    := true.B
    entry.hart_id  := hart_id
    entry.vbank_id := vbank_id
    entry.is_multi := is_multi
    entry.group_id := group_id
  }

  def deleteEntry(hart_id: UInt, vbank_id: UInt): Unit = {
    val found =
      mappingTable.map(entry => entry.valid && entry.vbank_id === vbank_id && entry.hart_id === hart_id).reduce(_ || _)
    when(!found) {
      assert(false.B, "SharedMemBackend release missing allocation: hart=%d vbank=%d\n", hart_id, vbank_id)
    }
    clearVbank(hart_id, vbank_id)
  }

  def clearVbank(hart_id: UInt, vbank_id: UInt): Unit = {
    for (i <- 0 until totalBanks) {
      when(mappingTable(i).valid && mappingTable(i).vbank_id === vbank_id && mappingTable(i).hart_id === hart_id) {
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
    accPipes(i).io.mem_req.is_shared := io.mem_req(i).is_shared
    accPipes(i).io.mem_req.hart_id   := io.mem_req(i).hart_id
    accPipes(i).io.mem_req.rob_id    := io.mem_req(i).rob_id
    accPipes(i).io.mem_req.inst_id   := io.mem_req(i).inst_id

    // Bank-side defaults (only driven when a bank is actually connected)
    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare
  }

  banks.zipWithIndex.foreach {
    case (bank, _) =>
      bank.io.sramRead.req.valid  := false.B
      bank.io.sramRead.req.bits   := DontCare
      bank.io.sramRead.resp.ready := true.B

      bank.io.sramWrite.req.valid  := false.B
      bank.io.sramWrite.req.bits   := DontCare
      bank.io.sramWrite.resp.ready := true.B
      bank.io.clear                := false.B
  }

  val realloc = io.config.bits.group_id === 0.U

  val freePbankMask = VecInit(mappingTable.map(entry =>
    !entry.valid ||
      (realloc && entry.hart_id === io.config.bits.hart_id && entry.vbank_id === io.config.bits.vbank_id)
  ))

  val hasFreePbank = freePbankMask.asUInt.orR
  io.config.ready := !io.config.bits.alloc || hasFreePbank
  when(io.config.valid && io.config.bits.alloc && !hasFreePbank) {
    assert(false.B, "SharedMemBackend allocation failed: no free physical shared bank\n")
  }

  // -----------------------------------------------------------------------------
  // Bank Alloc/Release
  // -----------------------------------------------------------------------------

  if (!useMesh) {
    hashMonitors.foreach { hashes =>
      for (j <- 0 until totalBanks) {
        hashes(j).io.write.valid     := false.B
        hashes(j).io.write.bits.addr := banks(j).io.sramWrite.req.bits.addr
        hashes(j).io.write.bits.mask := banks(j).io.sramWrite.req.bits.mask
        hashes(j).io.write.bits.data := banks(j).io.sramWrite.req.bits.data
      }
    }
  }

  hashMonitors.foreach(_.foreach(_.io.bind := false.B))

  when(io.config.fire) {
    when(io.config.bits.transfer) {
      val source      = io.config.bits.source_bank_id
      val target      = io.config.bits.vbank_id
      val owner       = io.config.bits.hart_id
      val sourceCount = PopCount(mappingTable.map(e => e.valid && e.hart_id === owner && e.vbank_id === source))
      val targetCount = PopCount(mappingTable.map(e => e.valid && e.hart_id === owner && e.vbank_id === target))
      val totalCount  = sourceCount +& targetCount
      assert(source =/= target && sourceCount =/= 0.U, "Shared bank transfer requires a distinct allocated source")
      assert(totalCount <= totalBanks.U, "Shared bank transfer exceeds physical bank count")
      for ((entry, physical) <- mappingTable.zipWithIndex) {
        when(entry.valid && entry.hart_id === owner && entry.vbank_id === source) {
          leases.foreach(counts => assert(counts(physical) === 0.U, "Shared bank transfer source has live exports"))
          entry.vbank_id := target
          entry.group_id := targetCount + entry.group_id
          entry.is_multi := totalCount > 1.U
        }
        when(entry.valid && entry.hart_id === owner && entry.vbank_id === target) {
          entry.is_multi := totalCount > 1.U
        }
      }
    }.elsewhen(io.config.bits.alloc) {
      when(io.config.bits.group_id === 0.U) {
        clearVbank(io.config.bits.hart_id, io.config.bits.vbank_id)
      }
      val pbankId = PriorityEncoder(freePbankMask)
      hashMonitors.foreach { hashes =>
        for (i <- 0 until totalBanks) {
          hashes(i).io.bind := pbankId === i.U
        }
      }
      printf(
        p"[SharedMemBackend][ALLOC] hart=${io.config.bits.hart_id} vbank=0x${Hexadecimal(io.config.bits.vbank_id)} " +
          p"group=${io.config.bits.group_id} pbank=$pbankId is_multi=${io.config.bits.is_multi}\n"
      )
      addEntry(
        io.config.bits.hart_id,
        io.config.bits.vbank_id,
        pbankId,
        io.config.bits.is_multi,
        io.config.bits.group_id
      )
    }.otherwise {
      printf(
        p"[SharedMemBackend][RELEASE] hart=${io.config.bits.hart_id} vbank=0x${Hexadecimal(io.config.bits.vbank_id)}\n"
      )
      deleteEntry(io.config.bits.hart_id, io.config.bits.vbank_id)
    }
  }

  // -----------------------------------------------------------------------------
  // Query interface: return group count for a given vbank_id
  // -----------------------------------------------------------------------------
  for (q <- 0 until nCores) {
    val queryValid  = RegNext(io.query_valid(q), false.B)
    val queryHart   = RegNext(io.query_hart_id(q), 0.U)
    val queryVbank  = RegNext(io.query_vbank_id(q), 0.U)
    val groupCounts = mappingTable.map { entry =>
      val matches = queryValid &&
        entry.valid &&
        (entry.hart_id === queryHart) &&
        (entry.vbank_id === queryVbank)
      val count   = Mux(entry.is_multi, entry.group_id +& 1.U, 1.U)
      Mux(matches, count, 0.U)
    }

    io.query_group_count(q) := RegNext(groupCounts.reduce((a, b) => Mux(a > b, a, b)), 0.U)
    val queryHit = groupCounts.map(_ =/= 0.U).reduce(_ || _)
    when(queryValid && !queryHit) {
      printf(
        p"[SharedMemBackend][QUERY_MISS] q=$q hart=$queryHart " +
          p"vbank=0x${Hexadecimal(queryVbank)} returned group_count=0\n"
      )
    }
  }

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  private def emitTrace(
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
    mtraces(ch).io.hart_id    := io.mem_req(ch).hart_id
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
    val activeHart  = Mux(accPipes(i).io.busy, accPipes(i).io.hart_id, io.mem_req(i).hart_id)
    val activeBank  = Mux(accPipes(i).io.busy, accPipes(i).io.bank_id, io.mem_req(i).bank_id)
    val activeGroup = Mux(accPipes(i).io.busy, accPipes(i).io.group_id, io.mem_req(i).group_id)
    val req_valid   = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid || accPipes(i).io.busy

    val tracePbankId = Wire(UInt(32.W))
    tracePbankId := 0.U
    for (j <- 0 until totalBanks) {
      val trace_hit_bank = mappingTable(j).valid &&
        (mappingTable(j).hart_id === activeHart) &&
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

    if (useMesh) {
      val channel  = mesh.get.io.channels(i)
      val mapped   = mappingTable.map(entry =>
        entry.valid &&
          entry.hart_id === activeHart &&
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
    } else {
      for (j <- 0 until totalBanks) {
        val hit_bank = mappingTable(j).valid &&
          (mappingTable(j).hart_id === activeHart) &&
          (mappingTable(j).vbank_id === activeBank) &&
          (!mappingTable(j).is_multi ||
            (mappingTable(j).is_multi && (mappingTable(j).group_id === activeGroup)))

        when(hit_bank && req_valid) {
          banks(j).io.sramRead <> accPipes(i).io.sramRead
          banks(j).io.sramWrite <> accPipes(i).io.sramWrite
          hashMonitors.foreach { hashes =>
            val monitor  = hashes(j)
            val physical = banks(j).io.sramWrite.req
            val offered  = accPipes(i).io.sramWrite.req
            physical.valid         := offered.valid && monitor.io.write.ready
            monitor.io.write.valid := offered.valid && physical.ready
            offered.ready          := physical.ready && monitor.io.write.ready
          }
        }
      }
    }
  }

  mesh.foreach(_.io.bankWrites.foreach(_.ready := true.B))

  hashMonitors.foreach { hashes =>
    for (j <- 0 until totalBanks) {
      val monitor = hashes(j)
      if (useMesh) {
        val write = mesh.get.io.bankWrites(j)
        write.ready                := monitor.io.write.ready
        monitor.io.write.valid     := write.valid
        monitor.io.write.bits.addr := write.bits.addr
        monitor.io.write.bits.mask := write.bits.mask.asBools
        monitor.io.write.bits.data := write.bits.data

      }
    }
  }

  io.bank_hashes.foreach { states =>
    for (i <- 0 until totalBanks) {
      states(i).valid      := mappingTable(i).valid
      states(i).hartId     := mappingTable(i).hart_id
      states(i).vbankId    := mappingTable(i).vbank_id
      states(i).pbankId    := i.U
      states(i).groupId    := mappingTable(i).group_id
      states(i).statusHash := hashMonitors.get(i).io.statusHash
    }
  }
}
