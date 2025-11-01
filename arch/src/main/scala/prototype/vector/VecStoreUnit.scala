package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.BuckyBallConfigs.CustomBuckyBallConfig


class ctrl_st_req(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val wr_bank = UInt(log2Up(b.sp_banks).W)
  val wr_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val iter = UInt(10.W)
}

class ex_st_req(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // Use accumulator type, 32 bits
  val rst = Vec(b.veclane, UInt(b.accType.getWidth.W))
  val iter = UInt(10.W)
}

class VecStoreUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new ctrl_st_req))
    val ex_st_i   = Flipped(Decoupled(new ex_st_req))

    // val sramWrite = Vec(b.sp_banks, new SramWriteIO(b.sp_bank_entries, spad_w, spad_w/8))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    val cmdResp_o = Valid(new Bundle {val commit = Bool()})
  })

	// val wr_bank 		 = RegInit(0.U(log2Up(b.sp_banks).W))
	val wr_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries).W))
  val iter 				 = RegInit(0.U(10.W))
  val iter_counter = RegInit(0.U(10.W))


  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
		// wr_bank 			:= io.ctrl_st_i.bits.wr_bank
		wr_bank_addr 	:= io.ctrl_st_i.bits.wr_bank_addr
    iter := (io.ctrl_st_i.bits.iter + 15.U(10.W)) & (~15.U(10.W))
		iter_counter 	:= 0.U
    state 		    := busy
  }

// -----------------------------------------------------------------------------
// Accept computation results from EX unit and perform write-back
// -----------------------------------------------------------------------------
	io.ex_st_i.ready := state === busy
  // for(i <- 0 until b.sp_banks) {
  //   io.sramWrite(i).en := false.B
  //   io.sramWrite(i).addr := 0.U
  //   io.sramWrite(i).data := 0.U
  //   io.sramWrite(i).mask := VecInit(Seq.fill(spad_w / 8)(false.B))
  // }
io.accWrite.foreach { acc =>
      acc.req.valid := false.B
      acc.req.bits.addr := 0.U
      acc.req.bits.data := Cat(Seq.fill(b.acc_w / 8)(0.U(8.W)))
      acc.req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(false.B))
    }
val waddr = wr_bank_addr + iter_counter(log2Ceil(b.veclane) - 1, 0)
	when(io.ex_st_i.fire) {
		for(i <- 0 until b.acc_banks/2) {
      when(waddr(0) === 0.U){
        io.accWrite(i).req.valid := true.B
			  io.accWrite(i).req.bits.addr :=  wr_bank_addr + (iter_counter(log2Ceil(b.veclane) - 1, 0) >> 1.U)

        // Each accumulator bank stores veclane/acc_banks elements
        // 16/4 = 4 elements
        val elementsPerBank = b.veclane / b.acc_banks * 2
        val startIdx = i * elementsPerBank
        val endIdx = startIdx + elementsPerBank - 1

        // Pack corresponding elements into a UInt
        val bankData = Cat(io.ex_st_i.bits.rst.slice(startIdx, endIdx + 1).reverse)
        io.accWrite(i).req.bits.data := bankData

        io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
		  }.otherwise{
        io.accWrite(i + b.acc_banks/2).req.valid := true.B
			  io.accWrite(i + b.acc_banks/2).req.bits.addr := wr_bank_addr + (iter_counter(log2Ceil(b.veclane) - 1, 0) >> 1.U)

        // Each accumulator bank stores veclane/acc_banks elements
        // 16/4 = 4 elements
        val elementsPerBank = b.veclane / b.acc_banks * 2
        val startIdx = i * elementsPerBank
        val endIdx = startIdx + elementsPerBank - 1

        // Pack corresponding elements into a UInt
        val bankData = Cat(io.ex_st_i.bits.rst.slice(startIdx, endIdx + 1).reverse)
        io.accWrite(i + b.acc_banks/2).req.bits.data := bankData

        io.accWrite(i + b.acc_banks/2).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
      }
    }
    iter_counter := iter_counter + 1.U
	}

// -----------------------------------------------------------------------------
// Reset iter counter, commit cmdResp, return to idle state
// -----------------------------------------------------------------------------
	when(state === busy && iter_counter >= iter) {
		state := idle
		io.cmdResp_o.valid := true.B
		io.cmdResp_o.bits.commit := true.B
	}.otherwise {
		io.cmdResp_o.valid := false.B
		io.cmdResp_o.bits.commit := false.B
  }



}
