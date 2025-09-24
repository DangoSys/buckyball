package examples.toy.balldomain

import chisel3._
import chisel3.util._
import framework.builtin.frontend.PostGDCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.DISA._
import framework.builtin.memdomain.dma.LocalAddr
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

// Ball域的详细解码输出
class BallDecodeCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // val is_vec        = Bool()
  // val is_bbfp       = Bool()
  // val is_im2col     = Bool()

  val bid = UInt(4.W) // Ball ID

  // 迭代次数
  val iter          = UInt(10.W)

  // Ball专用字段
  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  val special       = UInt(40.W) // 指令专用子段

  // Ball的操作数地址
  val op1_bank      = UInt(log2Up(b.sp_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val op2_bank      = UInt(log2Up(b.sp_banks).W)
  val op2_bank_addr = UInt(log2Up(b.spad_bank_entries).W)

  // 写入地址和bank信息
  val wr_bank       = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val wr_bank_addr  = UInt(log2Up(b.spad_bank_entries + b.acc_bank_entries).W)
  val is_acc        = Bool() // 是否是acc bank的操作

  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
}

// Ball decode fields
// VALID is used to indicate the command opcode is valid, invalid command will assert
object BallDecodeFields extends Enumeration {
  type Field = Value
  val OP1_EN, OP2_EN, WR_SPAD, OP1_FROM_SPAD, OP2_FROM_SPAD,
      OP1_SPADDR, OP2_SPADDR, WR_SPADDR, ITER, BID, SPECIAL, VALID = Value
}

object CtrlDecodeFields extends Enumeration {
  type Field = Value
  val FENCE_EN, VALID = Value
}


// Default constants for EX decoder
object BallDefaultConstants {
  val Y = true.B
  val N = false.B
  val DADDR = 0.U(14.W)
  val DITER = 0.U(10.W)
  val DBID = 0.U(4.W)
  val DSPECIAL = 0.U(40.W)
}

class BallDomainDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  import BallDefaultConstants._

  val io = IO(new Bundle {
    val raw_cmd_i = Flipped(Decoupled(new PostGDCmd))
    val ball_decode_cmd_o = Decoupled(new BallDecodeCmd)

    val fence_o = Output(Bool())
  })

  val spAddrLen = b.spAddrLen

  // 只处理ball指令
  io.raw_cmd_i.ready := io.ball_decode_cmd_o.ready

  val func7 = io.raw_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.raw_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.raw_cmd_i.bits.raw_cmd.rs2

  // ball指令解码
  import BallDecodeFields._
  val ball_default_decode = List(N,N,N,N,N,DADDR,DADDR,DADDR,DITER,DBID,DSPECIAL,N)
  val ball_decode_list = ListLookup(func7, ball_default_decode, Array(
    MATMUL_WARP16_BITPAT -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),1.U,rs2(63,spAddrLen + 10),Y),
    BB_BBFP_MUL          -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),2.U,rs2(63,spAddrLen + 10),Y),
    MATMUL_WS            -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),2.U,rs2(63,spAddrLen + 10),Y),
    IM2COL               -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),3.U,rs2(63,spAddrLen + 10),Y),
    TRANSPOSE            -> List(Y,Y,Y,Y,Y, rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen),4.U,rs2(63,spAddrLen + 10),Y)
  ))

  import CtrlDecodeFields._
  val ctrl_default_decode = List(N,N)
  val ctrl_decode_list = ListLookup(func7, ctrl_default_decode, Array(
    FENCE -> List(Y,Y)
  ))

  // 断言：解码列表中必须有VALID字段
  assert(!(io.raw_cmd_i.fire && !ball_decode_list(BallDecodeFields.VALID.id).asBool && !ctrl_decode_list(CtrlDecodeFields.VALID.id).asBool),
    "BallDomainDecoder: Invalid command opcode, func7 = 0x%x\n", func7)

// -----------------------------------------------------------------------------
// 输出赋值
// -----------------------------------------------------------------------------
  io.ball_decode_cmd_o.valid := io.raw_cmd_i.valid && io.raw_cmd_i.bits.is_ball && !(ctrl_decode_list(CtrlDecodeFields.VALID.id).asBool) // 需要不是控制信号才能进rob

  // io.ball_decode_cmd_o.bits.is_vec        := Mux(io.ball_decode_cmd_o.valid, func7 === MATMUL_WARP16_BITPAT,                        false.B)
  // io.ball_decode_cmd_o.bits.is_bbfp       := Mux(io.ball_decode_cmd_o.valid, func7 === BB_BBFP_MUL || func7 === MATMUL_WS,          false.B)
  // io.ball_decode_cmd_o.bits.is_im2col     := Mux(io.ball_decode_cmd_o.valid, func7 === IM2COL,                                      false.B)
  io.ball_decode_cmd_o.bits.bid           := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.BID.id).asUInt, DBID)

  io.ball_decode_cmd_o.bits.iter          := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.ITER.id).asUInt, 0.U(10.W))
  io.ball_decode_cmd_o.bits.special       := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.SPECIAL.id).asUInt, DSPECIAL)
  io.ball_decode_cmd_o.bits.op1_en        := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP1_EN.id).asBool,        false.B)
  io.ball_decode_cmd_o.bits.op2_en        := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP2_EN.id).asBool,        false.B)
  io.ball_decode_cmd_o.bits.wr_spad_en    := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.WR_SPAD.id).asBool,       false.B)
  io.ball_decode_cmd_o.bits.op1_from_spad := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP1_FROM_SPAD.id).asBool, false.B)
  io.ball_decode_cmd_o.bits.op2_from_spad := Mux(io.ball_decode_cmd_o.valid, ball_decode_list(BallDecodeFields.OP2_FROM_SPAD.id).asBool, false.B)

  // 地址解析
  val op1_spaddr = ball_decode_list(BallDecodeFields.OP1_SPADDR.id).asUInt
  val op2_spaddr = ball_decode_list(BallDecodeFields.OP2_SPADDR.id).asUInt
  val wr_spaddr  = ball_decode_list(BallDecodeFields.WR_SPADDR.id).asUInt

  val op1_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, op1_spaddr)
  val op2_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, op2_spaddr)
  val wr_laddr  = LocalAddr.cast_to_sp_addr(b.local_addr_t, wr_spaddr)

  io.ball_decode_cmd_o.bits.op1_bank      := Mux(io.ball_decode_cmd_o.valid, op1_laddr.sp_bank(), 0.U(log2Up(b.sp_banks).W))
  io.ball_decode_cmd_o.bits.op1_bank_addr := Mux(io.ball_decode_cmd_o.valid, op1_laddr.sp_row(),  0.U(log2Up(b.spad_bank_entries).W))
  io.ball_decode_cmd_o.bits.op2_bank      := Mux(io.ball_decode_cmd_o.valid, op2_laddr.sp_bank(), 0.U(log2Up(b.sp_banks).W))
  io.ball_decode_cmd_o.bits.op2_bank_addr := Mux(io.ball_decode_cmd_o.valid, op2_laddr.sp_row(),  0.U(log2Up(b.spad_bank_entries).W))

  io.ball_decode_cmd_o.bits.wr_bank       := Mux(io.ball_decode_cmd_o.valid, wr_laddr.mem_bank(), 0.U(log2Up(b.sp_banks + b.acc_banks).W))
  io.ball_decode_cmd_o.bits.wr_bank_addr  := Mux(io.ball_decode_cmd_o.valid, wr_laddr.mem_row(),  0.U(log2Up(b.spad_bank_entries + b.acc_bank_entries).W))
  io.ball_decode_cmd_o.bits.is_acc        := Mux(io.ball_decode_cmd_o.valid, (io.ball_decode_cmd_o.bits.wr_bank >= b.sp_banks.U), false.B)

  // 断言：执行指令中OpA和OpB必须访问不同的bank
  assert(!(io.ball_decode_cmd_o.valid && io.ball_decode_cmd_o.bits.op1_en && io.ball_decode_cmd_o.bits.op2_en &&
           io.ball_decode_cmd_o.bits.op1_bank === io.ball_decode_cmd_o.bits.op2_bank),
  "BallDomainDecoder: Ball instruction OpA and OpB cannot access the same bank")

// -----------------------------------------------------------------------------
// 继续传递rs1和rs2
// -----------------------------------------------------------------------------
  io.ball_decode_cmd_o.bits.rs1 := rs1
  io.ball_decode_cmd_o.bits.rs2 := rs2

// -----------------------------------------------------------------------------
// 控制信号不进入ROB
// -----------------------------------------------------------------------------
  val is_fence = ctrl_decode_list(CtrlDecodeFields.FENCE_EN.id).asBool

  // 当fence命令有效时，输出fence设置信号
  io.fence_o := io.raw_cmd_i.fire && is_fence
}
