package framework.acc.balldomain.decoder

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.balldomain.bbus.BallRequest

/** Selects the Ball that owns each funct7; BALL_INIT names its target in rs2[4:0]. */
@instantiable
class Decoder(fp: FrontendParam, mp: MemDomainParam, bp: BallDomainParam) extends Module {

  @public
  val io = IO(new Bundle {
    val issue   = Flipped(Decoupled(new BallRequest(fp, mp)))
    val command = Decoupled(new BallRequest(fp, mp))
  })

  val funct      = io.issue.bits.cmd.funct
  val init       = funct === DISA.BALL_INIT_BITPAT
  val target     = WireDefault(0.U(5.W))
  val registered = WireDefault(false.B)
  for {
    ball  <- bp.balls
    entry <- ball.instructions
  } {
    when(funct === entry.funct7.U) {
      target     := ball.id.U
      registered := true.B
    }
  }
  when(init) {
    target     := io.issue.bits.cmd.rs2(4, 0)
    registered := VecInit(bp.balls.map(b => target === b.id.U)).asUInt.orR
  }
  io.command.valid := io.issue.valid && registered
  io.command.bits        := io.issue.bits
  io.command.bits.target := target
  io.issue.ready         := io.command.ready && registered
  when(io.issue.valid) {
    assert(registered, "Ball command is not registered")
    when(init)(assert(io.issue.bits.cmd.rs2(63, 5) === 0.U, "BALL_INIT reserves rs2[63:5]"))
  }
}
