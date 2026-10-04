package framework.system.core.accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.top.GlobalConfig
import framework.frontend.Frontend
import framework.frontend.globalrs.RobAllocation
import hier.core.rocket.RobFault
import framework.system.core.rocket.{RoCCCommandBB, RoCCResponseBB}
import framework.memdomain.MemDomain
import framework.memdomain.backend.MemRequestIO
import framework.memdomain.backend.shared.SharedMemLayout
import framework.memdomain.frontend.mem.{Footprint, MemConfigerIO}
import framework.memdomain.frontend.mem.dma.DmaPort
import framework.balldomain.BallDomain
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.isa.{MvoverISA, MvoverPort}
import memcore.memory.mesh_shm.MeshLocalBankPort

/** Accelerator domains with an explicit DMA transport boundary owned by the enclosing system. */
@instantiable
class BuckyballAccelerator(val b: GlobalConfig) extends Module {
  val totalBallRead   = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite  = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val sharedHashCount = if (b.memDomain.sharedEnable) SharedMemLayout.totalBank(b) else 0

  @public
  val io = IO(new Bundle {
    // RoCC command/response (connected to Rocket core inside tile)
    val cmd                   = Flipped(Decoupled(new RoCCCommandBB(b.tile.xLen)))
    val resp                  = Decoupled(new RoCCResponseBB(b.tile.xLen))
    val busy                  = Output(Bool())
    val idle                  = Output(Bool())
    val interrupt             = Output(Bool())
    val hartid                = Input(UInt(b.tile.xLen.W))
    val sharedBankOwnerHartId = Input(UInt(b.tile.xLen.W))
    // Includes boot allocations; external context binds only on io.cmd.fire.
    val allocation            = Valid(new RobAllocation(b))
    val retired               = Output(UInt(b.frontend.rob_entries.W))
    val fault                 = Valid(new RobFault(b.frontend.rob_entries))
    val firstFault            = Output(Valid(new RobFault(b.frontend.rob_entries)))
    val footprints            = Output(Vec(3, new Footprint(b)))

    val dma = new DmaPort(b.memDomain.dma_buswidth)

    // Shared memory path — exposed to tile level for multi-core SharedMemBackend
    val shared_mem_req = Vec(SharedMemLayout.channelPerHart(b), new MemRequestIO(b))
    val mvover         = new MvoverPort

    val meshLocalBank = Flipped(new MeshLocalBankPort(
      MvoverISA.AddressBits,
      MvoverISA.BankBits,
      b.memDomain.bankWidth,
      math.max(1, log2Ceil(b.frontend.rob_entries))
    ))

    val shared_config            = Decoupled(new MemConfigerIO(b))
    val shared_query_valid       = Output(Bool())
    val shared_query_vbank_id    = Output(UInt(b.memDomain.vbankIdWidth.W))
    val shared_query_group_count = Input(UInt(b.memDomain.groupCountWidth.W))

    val shared_bank_hashes =
      if (b.sim.diffTest && b.memDomain.sharedEnable) {
        Some(Input(Vec(sharedHashCount, new PhysicalBankHash(b))))
      } else {
        None
      }

    // Barrier interface — connected to tile-level BarrierUnit
    val barrier_arrive  = Output(Bool())
    val barrier_release = Input(Bool())
  })

  // --- Instantiate domains ---
  val frontend:   Instance[Frontend]   = Instantiate(new Frontend(b))
  val ballDomain: Instance[BallDomain] = Instantiate(new BallDomain(b))
  val memDomain:  Instance[MemDomain]  = Instantiate(new MemDomain(b))
  frontend.io.hartid                        := io.hartid
  frontend.io.sharedBankOwnerHartId         := io.sharedBankOwnerHartId
  frontend.io.bank_hashes.foreach(_         := memDomain.io.bank_hashes.get)
  memDomain.io.shared_bank_hashes.foreach(_ := io.shared_bank_hashes.get)
  io.allocation                             := frontend.io.allocation
  io.retired                                := frontend.io.retired
  io.footprints                             := memDomain.io.footprints

  // --- Frontend <- cmd ---
  frontend.io.cmd.valid    := io.cmd.valid
  frontend.io.cmd.bits.cmd := io.cmd.bits
  io.cmd.ready             := frontend.io.cmd.ready

  // --- Frontend -> BallDomain ---
  ballDomain.global_issue_i <> frontend.io.ball_issue_o
  frontend.io.ball_complete_i <> ballDomain.global_complete_o

  // --- BallDomain -> Frontend (SubROB requests) ---
  for (i <- 0 until b.ballDomain.ballNum) {
    frontend.io.ball_subrob_req_i(i) <> ballDomain.subRobReq(i)
  }

  memDomain.io.ballChannelActive := ballDomain.ballChannelActive
  ballDomain.ballChannelReady    := memDomain.io.ballChannelReady

  // --- Frontend -> MemDomain ---
  memDomain.io.global_issue_i <> frontend.io.mem_issue_o
  frontend.io.mem_complete_i <> memDomain.io.global_complete_o
  // Observe the accepted completion before the scheduler's ID-only queue.
  // Its arbiter accepts at most one of these domain completions per cycle.
  val memoryFault     = memDomain.io.global_complete_o.fire && memDomain.io.global_complete_o.bits.fault.error =/= 0.U
  val ballFault       = ballDomain.global_complete_o.fire && ballDomain.global_complete_o.bits.fault.error =/= 0.U
  val faultCompletion = Mux(memoryFault, memDomain.io.global_complete_o.bits, ballDomain.global_complete_o.bits)
  io.fault.valid        := memoryFault || ballFault
  io.fault.bits.rob_id  := faultCompletion.rob_id
  io.fault.bits.error   := faultCompletion.fault.error
  io.fault.bits.address := faultCompletion.fault.address
  val firstFault = RegInit(0.U.asTypeOf(Valid(new RobFault(b.frontend.rob_entries))))
  when(io.fault.valid && !firstFault.valid)(firstFault := io.fault)
  io.firstFault                                        := firstFault
  memDomain.io.hartid                                  := io.hartid
  memDomain.io.inst_ids                                := frontend.io.inst_ids

  if (b.rvv.enable) {
    ballDomain.kernel_command_i.get <> memDomain.io.kernel_command.get
    memDomain.io.kernel_complete.get <> ballDomain.kernel_complete_o.get
    ballDomain.kernel.get <> memDomain.io.kernel.get
  }
  frontend.io.kernel_write_bank <> ballDomain.kernelWriteBank

  // --- BallDomain <-> MemDomain (bankRead with pipeline register to break comb loops) ---
  for (i <- 0 until totalBallRead) {
    val bankReadReqWithIds = Wire(Decoupled(new Bundle {
      val bank_id  = chiselTypeOf(ballDomain.bankRead(i).bank_id)
      val rob_id   = chiselTypeOf(ballDomain.bankRead(i).rob_id)
      val ball_id  = chiselTypeOf(ballDomain.bankRead(i).ball_id)
      val group_id = chiselTypeOf(ballDomain.bankRead(i).group_id)
      val req      = chiselTypeOf(ballDomain.bankRead(i).io.req.bits)
    }))

    bankReadReqWithIds.valid            := ballDomain.bankRead(i).io.req.valid
    bankReadReqWithIds.bits.bank_id     := ballDomain.bankRead(i).bank_id
    bankReadReqWithIds.bits.rob_id      := ballDomain.bankRead(i).rob_id
    bankReadReqWithIds.bits.ball_id     := ballDomain.bankRead(i).ball_id
    bankReadReqWithIds.bits.group_id    := ballDomain.bankRead(i).group_id
    bankReadReqWithIds.bits.req         := ballDomain.bankRead(i).io.req.bits
    ballDomain.bankRead(i).io.req.ready := bankReadReqWithIds.ready

    val bankReadReqQ = Queue(bankReadReqWithIds, 8)

    memDomain.io.ballDomain.bankRead(i).io.req.valid := bankReadReqQ.valid
    memDomain.io.ballDomain.bankRead(i).io.req.bits  := bankReadReqQ.bits.req
    memDomain.io.ballDomain.bankRead(i).bank_id      := bankReadReqQ.bits.bank_id
    memDomain.io.ballDomain.bankRead(i).rob_id       := bankReadReqQ.bits.rob_id
    memDomain.io.ballDomain.bankRead(i).ball_id      := bankReadReqQ.bits.ball_id
    memDomain.io.ballDomain.bankRead(i).group_id     := bankReadReqQ.bits.group_id
    bankReadReqQ.ready                               := memDomain.io.ballDomain.bankRead(i).io.req.ready

    ballDomain.bankRead(i).io.resp <> memDomain.io.ballDomain.bankRead(i).io.resp
  }

  ballDomain.bankWrite <> memDomain.io.ballDomain.bankWrite

  // --- BallDomain <-> MemDomain (MMIO read path) ---
  ballDomain.mmioRead <> memDomain.io.ballDomain.mmioRead
  ballDomain.mmioWrite <> memDomain.io.ballDomain.mmioWrite

  io.dma <> memDomain.io.dma

  // --- Shared memory passthrough ---
  io.shared_mem_req <> memDomain.io.shared_mem_req
  io.mvover <> memDomain.io.mvover
  io.meshLocalBank <> memDomain.io.meshLocalBank
  io.shared_config <> memDomain.io.shared_config
  io.shared_query_valid                 := memDomain.io.shared_query_valid
  io.shared_query_vbank_id              := memDomain.io.shared_query_vbank_id
  memDomain.io.shared_query_group_count := io.shared_query_group_count

  // --- Barrier passthrough ---
  io.barrier_arrive           := frontend.io.barrier_arrive
  frontend.io.barrier_release := io.barrier_release

  // --- Response & status ---
  io.resp <> frontend.io.resp
  io.busy      := frontend.io.busy
  io.idle      := frontend.io.idle && !io.dma.readBusy && !io.dma.writeBusy
  // Typed completion failures are delivered through the admission fault path.
  // firstFault is retained diagnostic state, not an additional sticky IRQ.
  io.interrupt := ballDomain.kernelFault

  // --- Busy watchdog ---
  // BootRom clears and initializes the local memories before the external
  // command interface becomes ready. That initialization can legitimately
  // exceed the runtime watchdog limit, so only start monitoring after the
  // first external command has handshaken.
  val runtime_started = RegInit(false.B)
  when(io.cmd.fire) {
    runtime_started := true.B
  }
  val busy_counter    = RegInit(0.U(32.W))
  busy_counter := Mux(runtime_started && frontend.io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000000.U, "BuckyballAccelerator: busy for too long!")
}
