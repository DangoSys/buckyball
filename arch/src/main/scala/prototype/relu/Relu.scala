package prototype.relu

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status
import prototype.relu.configs.ReluConfig

@instantiable
class PipelinedRelu(val parameter: ReluConfig)(implicit p: Parameters)
    extends Module
    with SerializableModule[ReluConfig] {
  // Get parameters from config
  val ballParam  = parameter.ballParam
  val InputNum   = parameter.InputNum
  val inputWidth = parameter.inputWidth
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    // cmd interface
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(ballParam)))
    val cmdResp = Decoupled(new BallRsComplete(ballParam))

    // Connect to unified bank read/write interface
    val bankRead =
      Vec(ballParam.numBanks, Flipped(new SramReadIO(ballParam.bankEntries, bankWidth)))

    val bankWrite = Vec(
      ballParam.numBanks,
      Flipped(new SramWriteIO(ballParam.bankEntries, bankWidth, ballParam.bankMaskLen))
    )

    // Status output
    val status = new Status
  })

  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state                                      = RegInit(idle)

  // Store a row of split elements (InputNum elements)
  // Store a InputNum x InputNum tile (perform element-wise ReLU then write back)
  val regArray = RegInit(
    VecInit(Seq.fill(InputNum)(
      VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    ))
  )

  // Counters
  val readCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val respCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(InputNum + 1).W))

  // Instruction registers
  val robid_reg = RegInit(0.U(10.W))
  val waddr_reg = RegInit(0.U(10.W))
  val wbank_reg = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val raddr_reg = RegInit(0.U(10.W))
  val rbank_reg = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val iter_reg  = RegInit(0.U(10.W))
  val cycle_reg = RegInit(0.U(6.W))
  // Batch iteration counter
  val iterCnt   = RegInit(0.U(32.W))

  // Precompute write data
  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(ballParam.bankMaskLen, UInt(1.W)))

  // SRAM default assignment
  for (i <- 0 until ballParam.numBanks) {
    io.bankRead(i).req.valid        := false.B
    io.bankRead(i).req.bits.addr    := 0.U
    io.bankRead(i).req.bits.fromDMA := false.B
    io.bankRead(i).resp.ready       := false.B

    io.bankWrite(i).req.valid     := false.B
    io.bankWrite(i).req.bits.addr := 0.U
    io.bankWrite(i).req.bits.data := 0.U
    io.bankWrite(i).req.bits.mask := VecInit(Seq.fill(ballParam.bankMaskLen)(0.U(1.W)))
  }

  // cmd interface default assignment
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := robid_reg

  // State machine
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state        := sRead
        readCounter  := 0.U
        respCounter  := 0.U
        writeCounter := 0.U

        robid_reg := io.cmdReq.bits.rob_id
        // For ReLU, output write-back should use decoded wr_bank/addr, not op2_* fields
        waddr_reg := 0.U // New ISA: all operations start from row 0
        wbank_reg := io.cmdReq.bits.cmd.wr_bank
        raddr_reg := 0.U // New ISA: all operations start from row 0
        rbank_reg := io.cmdReq.bits.cmd.op1_bank
        iter_reg  := io.cmdReq.bits.cmd.iter
        cycle_reg := (io.cmdReq.bits.cmd.iter +& (InputNum.U - 1.U)) / InputNum.U - 1.U
      }

      when(cycle_reg =/= 0.U) {
        state        := sRead
        readCounter  := 0.U
        writeCounter := 0.U
        respCounter  := 0.U
        waddr_reg    := waddr_reg + InputNum.U
        raddr_reg    := raddr_reg + InputNum.U
        cycle_reg    := cycle_reg - 1.U
      }
    }

    is(sRead) {
      when(readCounter < InputNum.U) {
        // Issue read request
        readCounter                          := readCounter + 1.U
        io.bankRead(rbank_reg).req.valid     := true.B
        io.bankRead(rbank_reg).req.bits.addr := raddr_reg + readCounter

      }

      // Receive response, only raise ready when there are outstanding reads
      val dataWord = io.bankRead(rbank_reg).resp.bits.data
      // val hasOutstandingRead = readCounter =/= respCounter
      io.bankRead(rbank_reg).resp.ready := true.B
      when(io.bankRead(rbank_reg).resp.fire) {
        for (col <- 0 until InputNum) {
          val hi     = (col + 1) * inputWidth - 1
          val lo     = col * inputWidth
          val raw    = dataWord(hi, lo)
          val signed = raw.asSInt
          val relu   = Mux(signed < 0.S, 0.S(inputWidth.W), signed)
          regArray(respCounter)(col) := relu.asUInt
        }
        respCounter := respCounter + 1.U
      }

      when(respCounter === InputNum.U) {
        state        := sWrite
        // Precompute first write data (row 0, concatenated by column)
        writeDataReg := Cat((0 until InputNum).reverse.map(j => regArray(0)(j)))
        // Set write mask (write all)
        for (i <- 0 until parameter.bankMaskLen) {
          writeMaskReg(i) := 1.U(1.W)
        }
      }
    }

    is(sWrite) {
      // Correctly use ready/valid handshake to advance writes, avoid dropped writes
      io.bankWrite(wbank_reg).req.valid     := writeCounter < InputNum.U
      io.bankWrite(wbank_reg).req.bits.addr := waddr_reg + writeCounter
      io.bankWrite(wbank_reg).req.bits.data := writeDataReg
      io.bankWrite(wbank_reg).req.bits.mask := writeMaskReg

      when(writeCounter === (InputNum - 1).U) {
        state := complete
      }.otherwise {
        writeCounter := writeCounter + 1.U
        // Prepare next row's write data
        writeDataReg := Cat((0 until InputNum).reverse.map(j => regArray(writeCounter + 1.U)(j)))
      }

    }

    is(complete) {
      when(cycle_reg === 0.U) {
        io.cmdResp.valid       := true.B
        io.cmdResp.bits.rob_id := robid_reg
        when(io.cmdResp.fire) {
          iterCnt := iterCnt + 1.U
        }
      }
      state := idle
    }
  }

  // Status signals
  io.status.ready    := io.cmdReq.ready
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := (state === idle)
  io.status.init     := (state === sRead) && (respCounter < InputNum.U)
  io.status.running  := (state === sWrite) || ((state === sRead) && (respCounter === InputNum.U))
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter     := iterCnt

  when(reset.asBool) {
    for (i <- 0 until InputNum) {
      for (j <- 0 until InputNum) {
        regArray(i)(j) := 0.U
      }
    }
    writeDataReg := 0.U
    for (i <- 0 until parameter.bankMaskLen) {
      writeMaskReg(i) := 0.U
    }
  }
}
