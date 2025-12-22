package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.backend.banks.SramWriteIO
import examples.toy.balldomain.BallDomainParam

class ctrl_st_req(parameter: BallDomainParam) extends Bundle {
  val wr_bank      = UInt(log2Up(parameter.numBanks).W)
  val wr_bank_addr = UInt(log2Up(parameter.bankEntries).W)
  val iter         = UInt(10.W)
}

class ex_st_req(parameter: BallDomainParam) extends Bundle {
  // Derived parameters
  val InputNum = 16
  val accWidth = 32
  // Use accumulator type, 32 bits
  val rst      = Vec(InputNum, UInt(accWidth.W))
  val iter     = UInt(10.W)
}

@instantiable
class VecStoreUnit(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum = 16
  val accWidth = 32

  @public
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new ctrl_st_req(parameter)))
    val ex_st_i   = Flipped(Decoupled(new ex_st_req(parameter)))

    // Unified bank write interface (writes use accumulate mode)
    val bankWrite =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, accWidth, parameter.bankMaskLen)))

    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  val wr_bank      = RegInit(0.U(log2Up(parameter.numBanks).W))
  val wr_bank_addr = RegInit(0.U(log2Up(parameter.bankEntries).W))
  val iter         = RegInit(0.U(10.W))
  val iter_counter = RegInit(0.U(10.W))

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter         := (io.ctrl_st_i.bits.iter + 15.U(10.W)) & (~15.U(10.W))
    iter_counter := 0.U
    state        := busy
  }

// -----------------------------------------------------------------------------
// Accept computation results from EX unit and perform write-back
// -----------------------------------------------------------------------------
  io.ex_st_i.ready := state === busy
  io.bankWrite.foreach { acc =>
    acc.req.valid     := false.B
    acc.req.bits.addr := 0.U
    acc.req.bits.data := Cat(Seq.fill(accWidth / 8)(0.U(8.W)))
    acc.req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(false.B))
  }
  val waddr = wr_bank_addr + iter_counter(log2Ceil(InputNum) - 1, 0)
  when(io.ex_st_i.fire) {
    for (i <- 0 until parameter.numBanks / 2) {
      when(waddr(0) === 0.U) {
        io.bankWrite(i).req.valid     := true.B
        io.bankWrite(i).req.bits.addr := wr_bank_addr + (iter_counter(log2Ceil(InputNum) - 1, 0) >> 1.U)

        // Each accumulator bank stores InputNum/numBanks elements
        val elementsPerBank = InputNum / parameter.numBanks * 2
        val startIdx        = i * elementsPerBank
        val endIdx          = startIdx + elementsPerBank - 1

        // Pack corresponding elements into a UInt
        val bankData = Cat(io.ex_st_i.bits.rst.slice(startIdx, endIdx + 1).reverse)
        io.bankWrite(i).req.bits.data := bankData

        io.bankWrite(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(true.B))
      }.otherwise {
        io.bankWrite(i + parameter.numBanks / 2).req.valid     := true.B
        io.bankWrite(i + parameter.numBanks / 2).req.bits.addr := wr_bank_addr + (iter_counter(
          log2Ceil(InputNum) - 1,
          0
        ) >> 1.U)

        // Each accumulator bank stores InputNum/numBanks elements
        val elementsPerBank = InputNum / parameter.numBanks * 2
        val startIdx        = i * elementsPerBank
        val endIdx          = startIdx + elementsPerBank - 1

        // Pack corresponding elements into a UInt
        val bankData = Cat(io.ex_st_i.bits.rst.slice(startIdx, endIdx + 1).reverse)
        io.bankWrite(i + parameter.numBanks / 2).req.bits.data := bankData

        io.bankWrite(i + parameter.numBanks / 2).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(true.B))
      }
    }
    iter_counter := iter_counter + 1.U
  }

// -----------------------------------------------------------------------------
// Reset iter counter, commit cmdResp, return to idle state
// -----------------------------------------------------------------------------
  when(state === busy && iter_counter >= iter) {
    state                    := idle
    io.cmdResp_o.valid       := true.B
    io.cmdResp_o.bits.commit := true.B
  }.otherwise {
    io.cmdResp_o.valid       := false.B
    io.cmdResp_o.bits.commit := false.B
  }

}
