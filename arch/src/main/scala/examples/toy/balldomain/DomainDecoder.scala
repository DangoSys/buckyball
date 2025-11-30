package examples.toy.balldomain

import chisel3._
import chisel3.util._
import framework.frontend.PostGDCmd
import examples.BuckyballConfigs.CustomBuckyballConfig
import examples.toy.balldomain.DISA._
import framework.memdomain.dma.LocalAddr
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

// Detailed decode output for Ball domain
class BallDecodeCmd(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  // Ball ID
  val bid = UInt(5.W)

  // Iteration count
  val iter          = UInt(10.W)

  // Ball-specific fields
  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  // Instruction-specific subfield
  val special       = UInt(40.W)

  // Ball operand addresses
  // 3 bits, supports 8 banks
  val op1_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  // 12 bits, uses SPAD row count
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  // 3 bits, supports 8 banks
  val op2_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  // 12 bits, uses SPAD row count
  val op2_bank_addr = UInt(log2Up(b.spad_bank_entries).W)

  // Write address and bank information
  // 3 bits, supports 8 banks
  val wr_bank       = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  // 12 bits, uses SPAD row count
  val wr_bank_addr  = UInt(log2Up(b.spad_bank_entries).W)
  // Whether this is an acc bank operation
  val is_acc        = Bool()

  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
}

// Ball decode fields
object BallDecodeFields extends Enumeration {
  type Field = Value
  val OP1_EN, OP2_EN, WR_SPAD, OP1_FROM_SPAD, OP2_FROM_SPAD,
      OP1_SPADDR, OP2_SPADDR, WR_SPADDR, ITER, BID, SPECIAL = Value
}



// Default constants for EX decoder
object BallDefaultConstants {
  val Y = true.B
  val N = false.B
  val DADDR = 0.U(14.W)
  val DITER = 0.U(10.W)
  val DBID = 0.U(5.W)
  val DSPECIAL = 0.U(40.W)
}

class BallDomainDecoder(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  import BallDefaultConstants._

  val io = IO(new Bundle {
    val raw_cmd_i = Flipped(Decoupled(new PostGDCmd))
    val ball_decode_cmd_o = Decoupled(new BallDecodeCmd)
  })

  val spAddrLen = b.spAddrLen

  // Only process ball instructions
  io.raw_cmd_i.ready := io.ball_decode_cmd_o.ready

  val func7 = io.raw_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.raw_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.raw_cmd_i.bits.raw_cmd.rs2

  // Ball instruction decoding
  import BallDecodeFields._
  val ball_default_decode = List(N,N,N,N,N,DADDR,DADDR,DADDR,DITER,DBID,DSPECIAL)
  val ball_decode_list = ListLookup(func7, ball_default_decode, Array(
    MATMUL_WARP16_BITPAT -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),0.U,rs2(63,spAddrLen + 10)),
    BB_BBFP_MUL          -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),1.U,rs2(63,spAddrLen + 10)),
    MATMUL_WS            -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),1.U,rs2(63,spAddrLen + 10)),
    IM2COL               -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),2.U,rs2(63,spAddrLen + 10)),
    TRANSPOSE            -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),3.U,rs2(63,spAddrLen + 10)),
    RELU                 -> List(Y,N,Y,Y,N, rs1(spAddrLen-1,0),                          DADDR, rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),4.U,rs2(63,spAddrLen + 10)),
    BBUS_CONFIG          -> List(Y,N,Y,Y,N, rs1(spAddrLen-1,0),                          DADDR, rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),5.U,rs2(63,spAddrLen + 10)),
    NNLUT                -> List(Y,N,Y,Y,N, rs1(spAddrLen-1,0),                          DADDR, rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),6.U,rs2(63,spAddrLen + 10)),
    SNN                  -> List(Y,N,Y,Y,N, rs1(spAddrLen-1,0),                          DADDR, rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),7.U,rs2(63,spAddrLen + 10)),
    ABFT_SYSTOLIC        -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),8.U,rs2(63,spAddrLen + 10)),
    CONV                 -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),9.U,rs2(63,spAddrLen + 10)),
    CIM                  -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),10.U,rs2(63,spAddrLen + 10)),
    TRANSFER              -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),11.U,rs2(63,spAddrLen + 10))
  ))

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  io.ball_decode_cmd_o.valid := io.raw_cmd_i.valid && io.raw_cmd_i.bits.is_ball

  io.ball_decode_cmd_o.bits.bid           := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.BID.id).asUInt, DBID)

  io.ball_decode_cmd_o.bits.iter          := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.ITER.id).asUInt,        0.U(10.W))
  io.ball_decode_cmd_o.bits.special       := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.SPECIAL.id).asUInt,      DSPECIAL)
  io.ball_decode_cmd_o.bits.op1_en        := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP1_EN.id).asBool,        false.B)
  io.ball_decode_cmd_o.bits.op2_en        := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP2_EN.id).asBool,        false.B)
  io.ball_decode_cmd_o.bits.wr_spad_en    := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.WR_SPAD.id).asBool,       false.B)
  io.ball_decode_cmd_o.bits.op1_from_spad := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP1_FROM_SPAD.id).asBool, false.B)
  io.ball_decode_cmd_o.bits.op2_from_spad := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP2_FROM_SPAD.id).asBool, false.B)

  // Address parsing
  val op1_spaddr = ball_decode_list(BallDecodeFields.OP1_SPADDR.id).asUInt
  val op2_spaddr = ball_decode_list(BallDecodeFields.OP2_SPADDR.id).asUInt
  val wr_spaddr  = ball_decode_list(BallDecodeFields.WR_SPADDR.id).asUInt

  val op1_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, op1_spaddr)
  val op2_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, op2_spaddr)
  val wr_laddr  = LocalAddr.cast_to_sp_addr(b.local_addr_t, wr_spaddr)

  // Use mem_bank() and mem_row() to support ACC banks (bank 4+)
  io.ball_decode_cmd_o.bits.op1_bank      := Mux(io.ball_decode_cmd_o.valid, op1_laddr.mem_bank(), 0.U(log2Up(b.sp_banks + b.acc_banks).W))
  io.ball_decode_cmd_o.bits.op1_bank_addr := Mux(io.ball_decode_cmd_o.valid, op1_laddr.mem_row(),  0.U(log2Up(b.spad_bank_entries).W))
  io.ball_decode_cmd_o.bits.op2_bank      := Mux(io.ball_decode_cmd_o.valid, op2_laddr.mem_bank(), 0.U(log2Up(b.sp_banks + b.acc_banks).W))
  io.ball_decode_cmd_o.bits.op2_bank_addr := Mux(io.ball_decode_cmd_o.valid, op2_laddr.mem_row(),  0.U(log2Up(b.spad_bank_entries).W))

  io.ball_decode_cmd_o.bits.wr_bank       := Mux(io.ball_decode_cmd_o.valid, wr_laddr.mem_bank(), 0.U(log2Up(b.sp_banks + b.acc_banks).W))
  io.ball_decode_cmd_o.bits.wr_bank_addr  := Mux(io.ball_decode_cmd_o.valid, wr_laddr.mem_row(),  0.U(log2Up(b.spad_bank_entries).W))
  io.ball_decode_cmd_o.bits.is_acc        := Mux(io.ball_decode_cmd_o.valid, (io.ball_decode_cmd_o.bits.wr_bank >= b.sp_banks.U), false.B)

  // Assertion: OpA and OpB in execution instructions must access different banks
  assert(!(io.ball_decode_cmd_o.valid && io.ball_decode_cmd_o.bits.op1_en && io.ball_decode_cmd_o.bits.op2_en &&
           io.ball_decode_cmd_o.bits.op1_bank === io.ball_decode_cmd_o.bits.op2_bank),
  "BallDomainDecoder: Ball instruction OpA and OpB cannot access the same bank")

// -----------------------------------------------------------------------------
// Continue passing rs1 and rs2
// -----------------------------------------------------------------------------
  io.ball_decode_cmd_o.bits.rs1 := rs1
  io.ball_decode_cmd_o.bits.rs2 := rs2
}
