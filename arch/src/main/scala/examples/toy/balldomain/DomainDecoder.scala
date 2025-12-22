package examples.toy.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import org.chipsalliance.cde.config.Parameters
import framework.frontend.decoder.{DomainId, PostGDCmd}
import examples.toy.balldomain.DISA._

// Detailed decode output for Ball domain
class BallDecodeCmd(parameter: BallDomainParam) extends Bundle {
  // Ball ID
  val bid = UInt(5.W)

  // Iteration count
  val iter = UInt(10.W)

  // Ball-specific fields
  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  // Instruction-specific subfield
  val special       = UInt(40.W)

  // Ball operand bank IDs (now directly from rs1/rs2)
  val op1_bank = UInt(log2Up(parameter.numBanks).W)
  val op2_bank = UInt(log2Up(parameter.numBanks).W)
  val wr_bank  = UInt(log2Up(parameter.numBanks).W)

  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
}

// Ball decode fields
object BallDecodeFields extends Enumeration {
  type Field = Value
  val OP1_EN, OP2_EN, WR_SPAD, OP1_FROM_SPAD, OP2_FROM_SPAD, OP1_SPADDR, OP2_SPADDR, WR_SPADDR, ITER, BID, SPECIAL =
    Value
}

// Default constants for EX decoder
object BallDefaultConstants {
  val Y        = true.B
  val N        = false.B
  val DADDR    = 0.U(14.W)
  val DITER    = 0.U(10.W)
  val DBID     = 0.U(5.W)
  val DSPECIAL = 0.U(40.W)
}

@instantiable
class BallDomainDecoder(val parameter: BallDomainParam)(implicit p: Parameters) extends Module {
  import BallDefaultConstants._

  @public
  val raw_cmd_i = IO(Flipped(Decoupled(new PostGDCmd)))

  @public
  val ball_decode_cmd_o = IO(Decoupled(new BallDecodeCmd(parameter)))

  val spAddrLen = 14 // b.spAddrLen replaced with constant

  // Only process ball instructions
  raw_cmd_i.ready := ball_decode_cmd_o.ready

  val func7 = raw_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = raw_cmd_i.bits.raw_cmd.rs1
  val rs2   = raw_cmd_i.bits.raw_cmd.rs2

  // Ball instruction decoding - now directly use rs1/rs2 as bank IDs
  import BallDecodeFields._
  val ball_default_decode = List(N, N, N, N, N, 0.U, 0.U, 0.U, DITER, DBID, DSPECIAL)

  val ball_decode_list = ListLookup(
    func7,
    ball_default_decode,
    Array(
      MATMUL_WARP16_BITPAT -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 0.U, rs2(63, 16)),
      BB_BBFP_MUL          -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 1.U, rs2(63, 16)),
      MATMUL_WS            -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 1.U, rs2(63, 16)),
      IM2COL               -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 2.U, rs2(63, 16)),
      TRANSPOSE            -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 3.U, rs2(63, 16)),
      RELU                 -> List(Y, N, Y, Y, N, rs1(7, 0), DADDR, rs2(7, 0), rs2(15, 8), 4.U, rs2(63, 16)),
      BBUS_CONFIG          -> List(Y, N, Y, Y, N, rs1(7, 0), DADDR, rs2(7, 0), rs2(15, 8), 5.U, rs2(63, 16)),
      NNLUT                -> List(Y, N, Y, Y, N, rs1(7, 0), DADDR, rs2(7, 0), rs2(15, 8), 31.U, rs2(63, 16)),
      SNN                  -> List(Y, N, Y, Y, N, rs1(7, 0), DADDR, rs2(7, 0), rs2(15, 8), 7.U, rs2(63, 16)),
      ABFT_SYSTOLIC        -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 8.U, rs2(63, 16)),
      CONV                 -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 9.U, rs2(63, 16)),
      CIM                  -> List(Y, Y, Y, Y, Y, rs1(7, 0), rs2(7, 0), rs1(15, 8), rs2(15, 8), 10.U, rs2(63, 16)),
      TRANSFER             -> List(Y, N, Y, Y, N, rs1(7, 0), DADDR, rs2(7, 0), rs2(15, 8), 6.U, rs2(63, 16))
    )
  )

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  ball_decode_cmd_o.valid := raw_cmd_i.valid && (raw_cmd_i.bits.domain_id === DomainId.BALL)

  ball_decode_cmd_o.bits.bid := Mux(ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.BID.id).asUInt, DBID)

  ball_decode_cmd_o.bits.iter          := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.ITER.id).asUInt,
    0.U(10.W)
  )
  ball_decode_cmd_o.bits.special       := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.SPECIAL.id).asUInt,
    DSPECIAL
  )
  ball_decode_cmd_o.bits.op1_en        := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.OP1_EN.id).asBool,
    false.B
  )
  ball_decode_cmd_o.bits.op2_en        := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.OP2_EN.id).asBool,
    false.B
  )
  ball_decode_cmd_o.bits.wr_spad_en    := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.WR_SPAD.id).asBool,
    false.B
  )
  ball_decode_cmd_o.bits.op1_from_spad := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.OP1_FROM_SPAD.id).asBool,
    false.B
  )
  ball_decode_cmd_o.bits.op2_from_spad := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.OP2_FROM_SPAD.id).asBool,
    false.B
  )

  // Directly assign bank IDs from decoded values
  ball_decode_cmd_o.bits.op1_bank := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.OP1_SPADDR.id).asUInt,
    0.U
  )
  ball_decode_cmd_o.bits.op2_bank := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.OP2_SPADDR.id).asUInt,
    0.U
  )
  ball_decode_cmd_o.bits.wr_bank  := Mux(
    ball_decode_cmd_o.valid,
    ball_decode_list(BallDecodeFields.WR_SPADDR.id).asUInt,
    0.U
  )

  // Assertion: OpA and OpB in execution instructions must access different banks
  assert(
    !(ball_decode_cmd_o.valid && ball_decode_cmd_o.bits.op1_en && ball_decode_cmd_o.bits.op2_en &&
      ball_decode_cmd_o.bits.op1_bank === ball_decode_cmd_o.bits.op2_bank),
    "BallDomainDecoder: Ball instruction OpA and OpB cannot access the same bank"
  )

// -----------------------------------------------------------------------------
// Continue passing rs1 and rs2
// -----------------------------------------------------------------------------
  ball_decode_cmd_o.bits.rs1 := rs1
  ball_decode_cmd_o.bits.rs2 := rs2
}
