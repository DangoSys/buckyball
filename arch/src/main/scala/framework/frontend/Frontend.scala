package framework.frontend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.frontend.boot.BootRom
import framework.frontend.decoder.GlobalDecoder
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue, GlobalScheduler, RobAllocation}
import framework.top.GlobalConfig
import framework.system.core.rocket.{RoCCCommandBB, RoCCResponseBB}
import framework.balldomain.blink.SubRobRow
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.backend.shared.SharedMemLayout

/**
 * Frontend Module
 * Encapsulates GlobalDecoder and global scheduler
 */
@instantiable
class Frontend(val b: GlobalConfig) extends Module {

  val sharedHashCount = if (b.memDomain.sharedEnable) SharedMemLayout.totalBank(b) else 0

  @public
  val io = IO(new Bundle {
    val hartid                = Input(UInt(b.tile.xLen.W))
    val sharedBankOwnerHartId = Input(UInt(b.tile.xLen.W))

    // RoCC command input
    val cmd = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommandBB(b.tile.xLen)
    }))

    // Issue to domains
    val ball_issue_o      = Decoupled(new GlobalSchedIssue(b))
    val mem_issue_o       = Decoupled(new GlobalSchedIssue(b))
    // Complete from domains
    val ball_complete_i   = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val mem_complete_i    = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val kernel_issue_o    = if (b.rvv.enable) Some(Decoupled(new GlobalSchedIssue(b))) else None
    val kernel_complete_i = if (b.rvv.enable) Some(Flipped(Decoupled(new GlobalSchedComplete(b)))) else None

    // Ball -> SubROB request passthrough
    val ball_subrob_req_i = Flipped(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))
    val inst_ids          = Output(Vec(b.frontend.rob_entries, UInt(64.W)))
    val allocation        = Valid(new RobAllocation(b))
    val retired           = Output(UInt(b.frontend.rob_entries.W))

    val bank_hashes =
      if (b.sim.diffTest) {
        Some(Input(Vec(b.memDomain.bankNum + sharedHashCount, new PhysicalBankHash(b))))
      } else {
        None
      }

    // RoCC response
    val resp = Decoupled(new RoCCResponseBB(b.tile.xLen))
    val busy = Output(Bool())
    val idle = Output(Bool())
    // Propagates the Global ROB retirement pulse to the host bridge.

    // Barrier interface — passthrough to GlobalRS
    val barrier_arrive  = Output(Bool())
    val barrier_release = Input(Bool())
  })

  val gDecoder:  Instance[GlobalDecoder]   = Instantiate(new GlobalDecoder(b))
  val scheduler: Instance[GlobalScheduler] = Instantiate(new GlobalScheduler(b))

  val boot: Instance[BootRom] = Instantiate(new BootRom(b))

  boot.io.schedulerIdle := scheduler.io.idle
  boot.io.cmd.ready     := boot.io.active && gDecoder.io.id_i.ready

  gDecoder.io.id_i.valid    := Mux(boot.io.active, boot.io.cmd.valid, io.cmd.valid)
  gDecoder.io.id_i.bits.cmd := Mux(boot.io.active, boot.io.cmd.bits.cmd, io.cmd.bits.cmd)
  // RoCC can't accept new instructions when boot is active
  io.cmd.ready              := !boot.io.active && gDecoder.io.id_i.ready

  scheduler.io.decode_cmd_i <> gDecoder.io.id_o
  scheduler.io.hart_id               := io.hartid
  scheduler.io.sharedBankOwnerHartId := io.sharedBankOwnerHartId
  scheduler.io.bank_hashes.foreach(_ := io.bank_hashes.get)

  io.ball_issue_o <> scheduler.io.ball_issue_o
  io.mem_issue_o <> scheduler.io.mem_issue_o
  if (b.rvv.enable) {
    io.kernel_issue_o.get <> scheduler.io.kernel_issue_o.get
    scheduler.io.kernel_complete_i.get <> io.kernel_complete_i.get
  }
  io.inst_ids   := scheduler.io.inst_ids
  io.allocation := scheduler.io.allocation
  io.retired    := scheduler.io.retired
  io.idle       := !boot.io.active && scheduler.io.idle && !gDecoder.io.id_o.valid

  scheduler.io.ball_complete_i <> io.ball_complete_i
  scheduler.io.mem_complete_i <> io.mem_complete_i

  // Wire SubROB request from BallDomain through to scheduler
  for (i <- 0 until b.ballDomain.ballNum) {
    scheduler.io.ball_subrob_req_i(i) <> io.ball_subrob_req_i(i)
  }

  io.resp <> scheduler.io.scheduler_rocc_o.resp

  // io.busy = 1 means NPU will block CPU
  // This is the only time when NPU will block CPU: RoB can't accept new
  // instructions (like meet fence, RoB full, barrier)
  //
  // Why we add boot.io.active here is because when boot is active, RoB is
  // typically full with mset instructions. This situation is not need to
  // block CPU.
  io.busy := !boot.io.active && scheduler.io.scheduler_rocc_o.busy

  // Barrier passthrough
  io.barrier_arrive            := scheduler.io.barrier_arrive
  scheduler.io.barrier_release := io.barrier_release

}
