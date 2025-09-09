package prototype
import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import firrtl2.passes.CheckTypes.st


class Im2col(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    // cmd接口
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // 连接到Scratchpad的SRAM读写接口
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))

  })

  val idle :: read :: read_and_convert :: complete :: Nil = Enum(4) // 状态定义
  val state = RegInit(idle)                         // 当前状态寄存器
  val ConvertBuffer = RegInit(VecInit(Seq.fill(4)(VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))))) //转换缓冲区
  val rowptr = RegInit(0.U(10.W))                   // 标志卷积窗口左上角的行指针
  val colptr = RegInit(0.U(5.W))                    // 标志卷积窗口左上角的列指针
  val reqcounter = RegInit(0.U(5.W))                // read状态下的请求计数器
  val respcounter = RegInit(0.U(5.W))               // read状态下的响应计数器
  val robid_reg = RegInit(0.U(10.W))                // 保存当前指令的RoB ID
  val krow_reg = RegInit(0.U(log2Up(b.veclane).W))  // 保存卷积核的行数
  val kcol_reg = RegInit(0.U(log2Up(b.veclane).W))  // 保存卷积核的列数
  val inrow_reg = RegInit(0.U(10.W))                // 保存输入矩阵的行数
  val incol_reg = RegInit(0.U((log2Up(b.veclane) + 1).W)) // 保存输入矩阵的列数
  val startcol_reg = RegInit(0.U((log2Up(b.veclane) + 1).W)) // 保存起始列号
  val startrow_reg = RegInit(0.U(10.W))             // 保存起始行号
  val waddr_reg = RegInit(0.U(10.W))                // 保存写入的起始地址
  val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))// 保存写入的bank
  val raddr_reg = RegInit(0.U(10.W))                // 保存读取的起始地址
  val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))// 保存读取`的bank


  //SRAM默认赋值
  for(i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := true.B
    io.sramWrite(i).req.valid       := false.B
    io.sramWrite(i).req.bits.addr   := 0.U
    io.sramWrite(i).req.bits.data   := 0.U
    io.sramWrite(i).req.bits.mask   := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
  }
  //cmd接口默认赋值
  io.cmdReq.ready := true.B
  io.cmdResp.valid := false.B
  io.cmdResp.bits.rob_id := 0.U

  val rowcnt = rowptr - startrow_reg
  val colcnt = colptr - startcol_reg
  val rowmax = inrow_reg - krow_reg
  val colmax = incol_reg - kcol_reg

  switch(state) {
    // 空闲状态，等待指令
    is(idle) {
      //指令到达，初始化各个寄存器
      when(io.cmdReq.fire) {
        state      := read
        rowptr     := io.cmdReq.bits.cmd.special(37,28)
        colptr     := io.cmdReq.bits.cmd.special(27,23)
        reqcounter := 0.U
        respcounter:= 0.U
        kcol_reg   := io.cmdReq.bits.cmd.special(3,0) // 卷积核列数
        krow_reg   := io.cmdReq.bits.cmd.special(7,4) // 卷积核行数
        incol_reg  := io.cmdReq.bits.cmd.special(12,8) // 输入矩阵列数
        inrow_reg  := io.cmdReq.bits.cmd.special(22,13) // 输入矩阵行数
        startcol_reg := io.cmdReq.bits.cmd.special(27,23) // 起始列号
        startrow_reg := io.cmdReq.bits.cmd.special(37,28) // 起始行号
        robid_reg  := io.cmdReq.bits.rob_id
        waddr_reg  := io.cmdReq.bits.cmd.op2_bank_addr
        wbank_reg  := io.cmdReq.bits.cmd.op2_bank
        raddr_reg  := io.cmdReq.bits.cmd.op1_bank_addr
        rbank_reg  := io.cmdReq.bits.cmd.op1_bank
      }
    }
    //读取一部分数据，填充ConvertBuffer
    is(read) {
      //发送读请求
      when(reqcounter < krow_reg) {
        reqcounter                           := reqcounter + 1.U
        io.sramRead(rbank_reg).req.valid     := true.B
        io.sramRead(rbank_reg).req.bits.addr := raddr_reg + reqcounter + startrow_reg
      }
      //处理读响应并存储在ConvertBuffer中
      when(io.sramRead(rbank_reg).resp.fire) {
        ConvertBuffer(respcounter)           := io.sramRead(rbank_reg).resp.bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
        respcounter                          := respcounter + 1.U
      }
      // 判断是否跳转状态
      state := Mux(respcounter === krow_reg, read_and_convert, read)

    }
    // 转换数据并读取剩余数据，写回spad
    is(read_and_convert) {
      // 移动指针
      when(colptr <= colmax && rowptr <= rowmax) {
        colptr := Mux(colptr === colmax, startcol_reg, colptr + 1.U)
        io.sramWrite(wbank_reg).req.valid     := true.B
        io.sramWrite(wbank_reg).req.bits.addr := waddr_reg + rowcnt * (colmax + 1.U - startcol_reg) + colcnt
        io.sramWrite(wbank_reg).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(~0.U(1.W)))
        io.sramWrite(wbank_reg).req.bits.data := {

          val window = Wire(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
          for (i <- 0 until b.veclane) {
              window(i) := 0.U  // 先全部初始化为0
          }

          // 填充窗口数据
          for (i <- 0 until 4; j <- 0 until 4) {
            when(i.U < krow_reg && j.U < kcol_reg) {
              val bufferRow = (rowcnt + i.U) % krow_reg
              val bufferCol = (colptr + j.U) % incol_reg
              window((i.U * kcol_reg) + j.U) := ConvertBuffer(bufferRow)(bufferCol)
            }.otherwise {
              window((i.U * kcol_reg) + j.U) := 0.U
            }
          }

          // 重新排列数据
          // 例如，对于klen_reg=3，将(00)(01)(02)(10)(11)(12)(20)(21)(22)组合
          Cat((0 until b.veclane).map(i => window(i)).reverse)
        }
      }
      //提前发送读请求
      when(colptr === colmax - 1.U){
        io.sramRead(rbank_reg).req.valid     := true.B
        io.sramRead(rbank_reg).req.bits.addr := raddr_reg + krow_reg + rowptr
      }
      //处理读响应并存储在ConvertBuffer中
      when(io.sramRead(rbank_reg).resp.fire){
        ConvertBuffer(rowcnt % krow_reg)     := io.sramRead(rbank_reg).resp.bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
        rowptr                               := rowptr + 1.U
      }
      // 判断是否跳转状态
      state := Mux(rowptr === rowmax && colptr === colmax, complete, read_and_convert)
    }
    // 完成状态，发送完成信号
    is(complete) {
     io.cmdResp.valid       := true.B
     io.cmdResp.bits.rob_id := robid_reg
     state                  := idle
    }
  }
}
