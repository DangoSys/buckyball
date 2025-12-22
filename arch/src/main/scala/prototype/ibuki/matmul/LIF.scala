package prototype.ibuki.matmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status

@instantiable
class LIF(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum   = 16
  val inputWidth = 8
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    // cmd interface
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(parameter)))
    val cmdResp = Decoupled(new BallRsComplete(parameter))

    // Connect to Scratchpad SRAM read/write interface
    val sramRead  = Vec(parameter.numBanks, Flipped(new SramReadIO(parameter.bankEntries, bankWidth)))
    val sramWrite =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, bankWidth, parameter.bankMaskLen)))

    // Status output
    val status = new Status
  })

  // State definitions
  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state                                      = RegInit(idle)

  // Store a InputNum x InputNum tile
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
  val wbank_reg = RegInit(0.U(log2Up(parameter.numBanks).W))
  val raddr_reg = RegInit(0.U(10.W))
  val rbank_reg = RegInit(0.U(log2Up(parameter.numBanks).W))
  val iter_reg  = RegInit(0.U(10.W))
  val cycle_reg = RegInit(0.U(6.W))
  val iterCnt   = RegInit(0.U(32.W))

  // LIF parameters from special field
  // special[7:0] = threshold (8 bits)
  // special[15:8] = leak_factor (8 bits, represents leak rate)
  val threshold_reg   = RegInit(127.U(8.W)) // Default threshold
  val leak_factor_reg = RegInit(240.U(8.W)) // Default leak factor (240/256 ≈ 0.9375)

  // Precompute write data
  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(parameter.bankMaskLen, UInt(1.W)))

  // SRAM default assignment
  for (i <- 0 until parameter.numBanks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := false.B

    io.sramWrite(i).req.valid     := false.B
    io.sramWrite(i).req.bits.addr := 0.U
    io.sramWrite(i).req.bits.data := 0.U
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(0.U(1.W)))
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
        waddr_reg := 0.U // New ISA: all operations start from row 0
        wbank_reg := io.cmdReq.bits.cmd.wr_bank
        raddr_reg := 0.U // New ISA: all operations start from row 0
        rbank_reg := io.cmdReq.bits.cmd.op1_bank
        iter_reg  := io.cmdReq.bits.cmd.iter
        cycle_reg := (io.cmdReq.bits.cmd.iter +& (InputNum.U - 1.U)) / InputNum.U - 1.U

        // Extract LIF parameters from special field
        threshold_reg   := io.cmdReq.bits.cmd.special(7, 0)
        leak_factor_reg := io.cmdReq.bits.cmd.special(15, 8)
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
        io.sramRead(rbank_reg).req.valid     := true.B
        io.sramRead(rbank_reg).req.bits.addr := raddr_reg + readCounter
      }

      // Receive response and perform LIF neuron computation
      io.sramRead(rbank_reg).resp.ready := true.B
      when(io.sramRead(rbank_reg).resp.fire) {
        for (col <- 0 until InputNum) {
          val hi     = (col + 1) * inputWidth - 1
          val lo     = col * inputWidth
          val raw    = io.sramRead(rbank_reg).resp.bits.data(hi, lo)
          val signed = raw.asSInt

          // LIF neuron model:
          // 1. Leak: membrane_potential = membrane_potential * leak_factor / 256
          // 2. Integrate: (input is already in membrane_potential, so just apply leak)
          // 3. Fire: if membrane_potential >= threshold, output spike (threshold value), else output leaked potential

          // Apply leak (multiply by leak_factor, then divide by 256)
          // leak_factor is unsigned (0-255), representing leak rate
          // Convert to signed for multiplication, then shift right by 8
          val leak_factor_signed = leak_factor_reg.zext.asSInt
          val leaked             = (signed * leak_factor_signed) >> 8

          // Fire condition: if leaked >= threshold, output spike (threshold), else output leaked
          // For simplicity, we output the threshold value as spike, or the leaked value
          val result = Mux(
            leaked >= threshold_reg.asSInt,
            threshold_reg.asSInt,
            Mux(leaked < (-threshold_reg).asSInt, (-threshold_reg).asSInt, leaked)
          )

          regArray(respCounter)(col) := result.asUInt
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
      // Write back results
      io.sramWrite(wbank_reg).req.valid     := writeCounter < InputNum.U
      io.sramWrite(wbank_reg).req.bits.addr := waddr_reg + writeCounter
      io.sramWrite(wbank_reg).req.bits.data := writeDataReg
      io.sramWrite(wbank_reg).req.bits.mask := writeMaskReg

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
