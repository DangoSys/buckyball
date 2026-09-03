package examples.balls.maxpool

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class MaxPoolBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "MaxPoolBall")
    .getOrElse(throw new IllegalArgumentException("MaxPoolBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "MAXPOOL")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("MAXPOOL not found in ballISA"))

  require(b.memDomain.bankWidth == 128, "MaxPoolBall requires 128-bit bank rows")
  require(mapping.inBW == 1, "MaxPoolBall requires inBW=1")
  require(mapping.outBW == 1, "MaxPoolBall requires outBW=1")
  require((funct >> 4) == 3, "MAXPOOL must encode one read and one write")

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val idle :: scan :: readResp :: writeReq :: writeResp :: complete :: Nil = Enum(6)
  private val state                                                                = RegInit(idle)
  private val robId                                                                = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub                                                                = RegInit(false.B)
  private val subRobId                                                             = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val inputBank                                                            = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank                                                           = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iter                                                                 = RegInit(0.U(b.frontend.iter_len.W))
  private val inputSide                                                            = RegInit(0.U(8.W))
  private val outputSide                                                           = RegInit(0.U(8.W))
  private val kernel                                                               = RegInit(0.U(8.W))
  private val stride                                                               = RegInit(0.U(8.W))
  private val padding                                                              = RegInit(0.U(8.W))
  private val inputBase                                                            = RegInit(0.U(6.W))
  private val outputBase                                                           = RegInit(0.U(6.W))
  private val outputStride                                                         = RegInit(0.U(6.W))
  private val startRow                                                             = RegInit(0.U(4.W))
  private val startCol                                                             = RegInit(0.U(4.W))
  private val outputRow                                                            = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val outputY                                                              = RegInit(0.U(8.W))
  private val outputX                                                              = RegInit(0.U(8.W))
  private val kernelY                                                              = RegInit(0.U(8.W))
  private val kernelX                                                              = RegInit(0.U(8.W))
  private val maximum                                                              = RegInit(VecInit(Seq.fill(16)("h80".U(8.W))))
  private val writeData                                                            = Reg(UInt(128.W))

  private val sourceY       = outputY * stride +& kernelY +& startRow
  private val sourceX       = outputX * stride +& kernelX +& startCol
  private val sourceValid   = sourceY >= padding && sourceX >= padding &&
    sourceY < padding +& inputSide && sourceX < padding +& inputSide
  private val inputY        = sourceY - padding
  private val inputX        = sourceX - padding
  private val inputAddress  = inputBase +& inputY * inputSide +& inputX
  private val outputAddress = outputBase +& outputY * outputStride +& outputX
  private val lastKernel    = kernelY +& 1.U === kernel && kernelX +& 1.U === kernel

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  io.bankRead(0).rob_id           := robId
  io.bankRead(0).ball_id          := 0.U
  io.bankRead(0).bank_id          := inputBank
  io.bankRead(0).group_id         := 0.U
  io.bankRead(0).io.req.valid     := false.B
  io.bankRead(0).io.req.bits.addr := inputAddress
  io.bankRead(0).io.resp.ready    := false.B

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := 0.U
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := outputAddress
  io.bankWrite(0).io.req.bits.data := writeData
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd          = io.cmdReq.bits.cmd
        val inSide       = cmd.rs2(3, 0)
        val outSide      = cmd.rs2(7, 4)
        val kernelSize   = cmd.rs2(11, 8)
        val poolStride   = cmd.rs2(15, 12)
        val poolPadding  = cmd.rs2(19, 16)
        val inBase       = cmd.rs2(25, 20)
        val outBase      = cmd.rs2(31, 26)
        val outStride    = cmd.rs2(37, 32)
        val poolStartRow = cmd.rs2(41, 38)
        val poolStartCol = cmd.rs2(45, 42)
        assert(cmd.rs2(63, 46) === 0.U, "MAXPOOL reserved rs2 bits must be zero")
        assert(cmd.funct7 === funct.U, "MaxPoolBall funct7 must be MAXPOOL")
        assert(cmd.rs1(19, 10) === 0.U, "MAXPOOL reserves input bank 1")
        assert(
          cmd.rs1(9, 0) < b.memDomain.bankNum.U &&
            cmd.rs1(29, 20) < b.memDomain.bankNum.U,
          "MAXPOOL bank id is invalid"
        )
        assert(
          cmd.op1_col === 1.U && cmd.op2_col === 0.U && cmd.wr_col === 1.U,
          "MAXPOOL requires one input bank and one output bank"
        )
        assert(cmd.op1_bank =/= cmd.wr_bank, "MAXPOOL input and output banks must differ")
        assert(
          inSide > 0.U && outSide > 0.U && kernelSize > 0.U && poolStride > 0.U,
          "MAXPOOL dimensions must be positive"
        )
        assert(outStride >= outSide, "MAXPOOL output stride is too small")
        assert(inBase +& inSide * inSide <= b.memDomain.bankEntries.U, "MAXPOOL input tile exceeds bank depth")
        assert(
          outSide * outSide === cmd.iter &&
            outBase +& (outSide - 1.U) * outStride +& outSide <= b.memDomain.bankEntries.U,
          "MAXPOOL output tile exceeds bank depth"
        )
        assert(
          inSide +& (poolPadding << 1) >= kernelSize +& poolStartRow &&
            inSide +& (poolPadding << 1) - kernelSize - poolStartRow ===
            (outSide - 1.U) * poolStride &&
            inSide +& (poolPadding << 1) >= kernelSize +& poolStartCol &&
            inSide +& (poolPadding << 1) - kernelSize - poolStartCol ===
            (outSide - 1.U) * poolStride,
          "MAXPOOL pooling geometry is inconsistent"
        )

        robId        := io.cmdReq.bits.rob_id
        isSub        := io.cmdReq.bits.is_sub
        subRobId     := io.cmdReq.bits.sub_rob_id
        inputBank    := cmd.op1_bank
        outputBank   := cmd.wr_bank
        iter         := cmd.iter
        inputSide    := inSide
        outputSide   := outSide
        kernel       := kernelSize
        stride       := poolStride
        padding      := poolPadding
        inputBase    := inBase
        outputBase   := outBase
        outputStride := outStride
        startRow     := poolStartRow
        startCol     := poolStartCol
        outputRow    := 0.U
        outputY      := 0.U
        outputX      := 0.U
        kernelY      := 0.U
        kernelX      := 0.U
        maximum      := VecInit(Seq.fill(16)("h80".U(8.W)))
        state        := scan
      }
    }

    is(scan) {
      when(sourceValid) {
        io.bankRead(0).io.req.valid := true.B
        when(io.bankRead(0).io.req.fire) {
          state := readResp
        }
      }.elsewhen(lastKernel) {
        writeData := Cat(maximum.reverse)
        state     := writeReq
      }.otherwise {
        when(kernelX +& 1.U === kernel) {
          kernelX := 0.U
          kernelY := kernelY + 1.U
        }.otherwise {
          kernelX := kernelX + 1.U
        }
      }
    }

    is(readResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        val data        = io.bankRead(0).io.resp.bits.data.asTypeOf(Vec(16, UInt(8.W)))
        val nextMaximum = Wire(Vec(16, UInt(8.W)))
        for (lane <- 0 until 16)
          nextMaximum(lane) := Mux(data(lane).asSInt > maximum(lane).asSInt, data(lane), maximum(lane))
        maximum             := nextMaximum
        when(lastKernel) {
          writeData := Cat(nextMaximum.reverse)
          state     := writeReq
        }.otherwise {
          when(kernelX +& 1.U === kernel) {
            kernelX := 0.U
            kernelY := kernelY + 1.U
          }.otherwise {
            kernelX := kernelX + 1.U
          }
          state := scan
        }
      }
    }

    is(writeReq) {
      io.bankWrite(0).io.req.valid := true.B
      when(io.bankWrite(0).io.req.fire) {
        state := writeResp
      }
    }

    is(writeResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(outputRow +& 1.U === iter) {
          state := complete
        }.otherwise {
          outputRow := outputRow + 1.U
          when(outputX +& 1.U === outputSide) {
            outputX := 0.U
            outputY := outputY + 1.U
          }.otherwise {
            outputX := outputX + 1.U
          }
          kernelY   := 0.U
          kernelX   := 0.U
          maximum   := VecInit(Seq.fill(16)("h80".U(8.W)))
          state     := scan
        }
      }
    }

    is(complete) {
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }
}
