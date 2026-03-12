package examples.toy.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import org.chipsalliance.cde.config.Parameters
import framework.frontend.decoder.{DomainId, PostGDCmd}
import examples.toy.balldomain.DISA._
import framework.top.GlobalConfig

// Detailed decode output for Ball domain
class BallDecodeCmd(numBanks: Int, iterLen: Int) extends Bundle {
  val bid  = UInt(5.W)
  // Iteration count
  val iter = UInt(iterLen.W)

  // Ball-specific fields
  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  // Instruction-specific subfield (full rs2)
  val special       = UInt(64.W)

  // Ball operand bank IDs
  val op1_bank = UInt(log2Up(numBanks).W)
  val op2_bank = UInt(log2Up(numBanks).W)
  val wr_bank  = UInt(log2Up(numBanks).W)

  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
}

// Ball decode fields
object BallDecodeFields extends Enumeration {
  type Field = Value
  val OP1_EN, OP2_EN, WR_SPAD, OP1_FROM_SPAD, OP2_FROM_SPAD, OP1_SPADDR, OP2_SPADDR, WR_SPADDR, BID, SPECIAL =
    Value
}

// Default constants for EX decoder
object BallDefaultConstants {
  val Y        = true.B
  val N        = false.B
  val DADDR    = 0.U(15.W)
  val DBID     = 0.U(5.W)
  val DSPECIAL = 0.U(64.W)
}

@instantiable
class BallDomainDecoder(val b: GlobalConfig) extends Module {
  import BallDefaultConstants._

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

  // Unified rs1 layout:
  //   rs1[14:0]  = BANK0 (op1 bank)
  //   rs1[29:15] = BANK1 (op2 bank)
  //   rs1[44:30] = BANK2 (wr bank)
  //   rs1[45]    = BB_RD0
  //   rs1[46]    = BB_RD1
  //   rs1[47]    = BB_WR
  //   rs1[63:48] = ITER
  // rs2 = special (full 64-bit)

  val op1_bank_raw = rs1(bankIdLen - 1, 0)
  val op2_bank_raw = rs1(bankIdLen + 14, 15)
  val wr_bank_raw  = rs1(bankIdLen + 29, 30)
  val iter_raw     = rs1(63, 48)

  // Ball instruction decoding
  import BallDecodeFields._
  val ball_default_decode = List(N, N, N, N, N, 0.U, 0.U, 0.U, DBID, DSPECIAL)

  val ball_decode_list = ListLookup(
    func7,
    ball_default_decode,
    Array(
      // Unified encoding: banks from rs1[14:0]/[29:15]/[44:30], iter from rs1[63:48], special = rs2
      //                        op1 op2 wr  op1s op2s  op1_bank       op2_bank        wr_bank        bid  special
      // Gemmini systolic array instructions — all share bid=7, sub-command in special[3:0]
      MATMUL_WARP16               -> List(Y, Y, Y, Y, Y, op1_bank_raw, op2_bank_raw, wr_bank_raw, 0.U, rs2),
      RELU                        -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 1.U, rs2),
      TRANSPOSE                   -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 2.U, rs2),
      IM2COL                      -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 3.U, rs2),
      SYSTOLIC                    -> List(Y, Y, Y, Y, Y, op1_bank_raw, op2_bank_raw, wr_bank_raw, 4.U, rs2),
      QUANT                       -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 5.U, rs2),
      DEQUANT                     -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 6.U, rs2),
      GEMMINI_CONFIG              -> List(N, N, N, N, N, DADDR, DADDR, DADDR, 7.U, Cat(rs2(63, 4), 0.U(4.W))),
      GEMMINI_PRELOAD             -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 7.U, Cat(rs2(63, 4), 1.U(4.W))),
      GEMMINI_COMPUTE_PRELOADED   -> List(
        Y,
        Y,
        Y,
        Y,
        Y,
        op1_bank_raw,
        op2_bank_raw,
        wr_bank_raw,
        7.U,
        Cat(rs2(63, 4), 2.U(4.W))
      ),
      GEMMINI_COMPUTE_ACCUMULATED -> List(
        Y,
        Y,
        Y,
        Y,
        Y,
        op1_bank_raw,
        op2_bank_raw,
        wr_bank_raw,
        7.U,
        Cat(rs2(63, 4), 3.U(4.W))
      ),
      GEMMINI_FLUSH               -> List(N, N, N, N, N, DADDR, DADDR, DADDR, 7.U, Cat(0.U(60.W), 4.U(4.W))),
      // TraceBall instructions
      BDB_COUNTER                 -> List(N, N, N, N, N, DADDR, DADDR, DADDR, 8.U, rs2),
      BDB_BACKDOOR                -> List(Y, N, Y, Y, N, op1_bank_raw, DADDR, wr_bank_raw, 8.U, rs2)
    )
  )

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  ball_decode_cmd_o.valid := cmd_i.valid && (cmd_i.bits.domain_id === DomainId.BALL)

  ball_decode_cmd_o.bits.bid := Mux(ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.BID.id).asUInt, DBID)

  // iter is always from rs1[63:48]
  ball_decode_cmd_o.bits.iter          := Mux(
    ball_decode_cmd_o.valid,
    iter_raw,
    0.U(iterLen.W)
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
