package framework.balldomain.decoder

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.top.GlobalConfig

/**
 * Ball-domain instruction decoder.
 *
 *  - funct7 encoding contract → enable bits, operand/write banks, iter
 *    count, pass-through of rs1/rs2/funct7 (pure framework concern).
 *  - funct7 → bid routing, generated from `b.ballDomain.ballISA`
 *    (a `(mnemonic, funct7, bid)` table the user provides via
 *    `default.json`); no per-example Scala decoder is needed.
 */
@instantiable
class BallDomainDecoder(val b: GlobalConfig) extends Module {
  val bankIdLen = b.frontend.bank_id_len
  val iterLen   = b.frontend.iter_len

  @public
  val cmd_i             = IO(Flipped(Decoupled(new PostGDCmd(b))))
  @public
  val ball_decode_cmd_o = IO(Decoupled(new BallDecodeCmd(b.memDomain.bankNum, iterLen)))

  cmd_i.ready := ball_decode_cmd_o.ready

  val func7 = cmd_i.bits.cmd.funct
  val rs1   = cmd_i.bits.cmd.rs1Data
  val rs2   = cmd_i.bits.cmd.rs2Data

  val op1_bank_raw = rs1(bankIdLen - 1, 0)
  val op2_bank_raw = rs1(bankIdLen + 9, 10)
  val wr_bank_raw  = rs1(bankIdLen + 19, 20)
  val iter_raw     = rs1(63, 30)

  // funct7[6:4] enable encoding:
  //   000 no bank   001 1rd       010 1wr        011 1rd+1wr
  //   100 2rd+1wr   101/110/111 no bank (extended opcode space)
  val enableBits = func7(6, 4)
  val hasRd0     = enableBits === 1.U || enableBits === 3.U || enableBits === 4.U
  val hasRd1     = enableBits === 4.U
  val hasWr      = enableBits === 2.U || enableBits === 3.U || enableBits === 4.U

  // funct7 -> bid lookup, generated from b.ballDomain.ballISA
  val bidWire      = WireDefault(0.U(5.W))
  val isaValidWire = WireDefault(false.B)
  b.ballDomain.ballISA.foreach { e =>
    when(func7 === e.funct7.U(7.W)) {
      bidWire      := e.bid.U
      isaValidWire := true.B
    }
  }

  ball_decode_cmd_o.valid              := cmd_i.valid && (cmd_i.bits.domain_id === DomainId.BALL)
  ball_decode_cmd_o.bits.bid           := Mux(ball_decode_cmd_o.valid, bidWire, 0.U)
  ball_decode_cmd_o.bits.funct7        := Mux(ball_decode_cmd_o.valid, func7, 0.U)
  ball_decode_cmd_o.bits.iter          := Mux(ball_decode_cmd_o.valid, iter_raw, 0.U(iterLen.W))
  ball_decode_cmd_o.bits.special       := Mux(ball_decode_cmd_o.valid, rs2, 0.U(64.W))
  ball_decode_cmd_o.bits.op1_bank      := Mux(ball_decode_cmd_o.valid && hasRd0, op1_bank_raw, 0.U)
  ball_decode_cmd_o.bits.op2_bank      := Mux(ball_decode_cmd_o.valid && hasRd1, op2_bank_raw, 0.U)
  ball_decode_cmd_o.bits.wr_bank       := Mux(ball_decode_cmd_o.valid && hasWr, wr_bank_raw, 0.U)
  ball_decode_cmd_o.bits.op1_en        := ball_decode_cmd_o.valid && hasRd0
  ball_decode_cmd_o.bits.op2_en        := ball_decode_cmd_o.valid && hasRd1
  ball_decode_cmd_o.bits.wr_spad_en    := ball_decode_cmd_o.valid && hasWr
  ball_decode_cmd_o.bits.op1_from_spad := ball_decode_cmd_o.valid && hasRd0
  ball_decode_cmd_o.bits.op2_from_spad := ball_decode_cmd_o.valid && hasRd1

  ball_decode_cmd_o.bits.rs1 := rs1
  ball_decode_cmd_o.bits.rs2 := rs2

  assert(
    !(cmd_i.fire && (cmd_i.bits.domain_id === DomainId.BALL) && !isaValidWire),
    "BallDomainDecoder: funct7 has no matching entry in ballISA"
  )
  assert(
    !(ball_decode_cmd_o.valid && ball_decode_cmd_o.bits.op1_en && ball_decode_cmd_o.bits.op2_en &&
      ball_decode_cmd_o.bits.op1_bank === ball_decode_cmd_o.bits.op2_bank),
    "BallDomainDecoder: Ball instruction OpA and OpB cannot access the same bank"
  )
}
