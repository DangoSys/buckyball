package framework.memdomain.mem

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._

import framework.builtin.util.Util._
import examples.BuckyballConfigs.CustomBuckyballConfig

import framework.memdomain.mem.AccMonitor
import framework.balldomain.blink.{SramReadWithInfo, SramWriteWithInfo}


class Scratchpad(implicit b: CustomBuckyballConfig, implicit val p: Parameters)
  extends Module with HasCoreParameters {

  // Assertion: ensure configuration consistency
  assert(b.sp_singleported, "Scratchpad expects single-ported SRAM banks")
  private val numBanks = b.sp_banks + b.acc_banks
  val io = IO(new Bundle {
    // SRAM read/write interface - used by load/store
    val dma = new Bundle {
      val sramread  = Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w))
      val sramwrite = Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
    }
    // Execution unit read/write interface - one read and write per bank, OpA and OpB guaranteed to access different banks
    val exec = new Bundle {
      val sramread  = Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w))
      val sramwrite = Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
    }
  })

// -----------------------------------------------------------------------------
// Scratchpad
// -----------------------------------------------------------------------------

  // SRAM banks - each bank has only one port, supports simultaneous read and write
  val spad_mems = Seq.fill(numBanks) { Module(new AccMonitor(
    b.spad_bank_entries, b.spad_w,
    // Use configuration parameters
    b.aligned_to, b.sp_singleported
  )) }

  // Request arbitration and connection for each bank
  spad_mems.zipWithIndex.foreach { case (bank, i) =>

    // All read request sources
    val main_read_req = io.dma.sramread(i).io.req
    val exec_read_req = io.exec.sramread(i).io.req

    // All write request sources
    val main_write = io.dma.sramwrite(i)
    val exec_write = io.exec.sramwrite(i)

    // Assertion: OpA and OpB should not access the same bank simultaneously
    assert(!(exec_read_req.valid && exec_write.io.req.valid),
      s"Bank ${i}: exec and write cannot access the same bank simultaneously")

    // Read request arbitration: priority exec > main
    val exec_read_sel = exec_read_req.valid
    val main_read_sel = main_read_req.valid && !exec_read_sel

    // Write request arbitration: exec has higher priority
    val exec_write_sel = exec_write.io.req.valid

    val main_isacc_sel = main_write.is_acc
    val exec_isacc_sel = exec_write.is_acc

    val main_bankid = main_write.bank_id
    val exec_bankid = exec_write.bank_id

    val main_robid = main_write.rob_id
    val exec_robid = exec_write.rob_id

    // Connect read request to SramBank
    bank.io.read.io.req.valid := exec_read_sel || main_read_sel
    bank.io.read.io.req.bits := Mux(exec_read_sel, exec_read_req.bits, main_read_req.bits)

    // Read request ready reverse connection
    main_read_req.ready := main_read_sel && bank.io.read.io.req.ready
    exec_read_req.ready := exec_read_sel && bank.io.read.io.req.ready

    // Record which client initiated read request (for response distribution)
    val resp_to_main = RegNext(main_read_sel && bank.io.read.io.req.fire, false.B)
    val resp_to_exec = RegNext(exec_read_sel && bank.io.read.io.req.fire, false.B)

    // Read response distribution
    io.dma.sramread(i).io.resp.valid := bank.io.read.io.resp.valid && resp_to_main
    io.dma.sramread(i).io.resp.bits := bank.io.read.io.resp.bits

    io.exec.sramread(i).io.resp.valid := bank.io.read.io.resp.valid && resp_to_exec
    io.exec.sramread(i).io.resp.bits := bank.io.read.io.resp.bits

    // Read response ready signal: either client ready is sufficient
    bank.io.read.io.resp.ready :=
      (resp_to_main && io.dma.sramread(i).io.resp.ready) ||
      (resp_to_exec && io.exec.sramread(i).io.resp.ready)

    // Connect write request to SramBank
    bank.io.write.io.req.valid     := Mux(exec_write_sel, exec_write.io.req.valid, main_write.io.req.valid)
    bank.io.write.io.req.bits.addr := Mux(exec_write_sel, exec_write.io.req.bits.addr, main_write.io.req.bits.addr)
    bank.io.write.io.req.bits.data := Mux(exec_write_sel, exec_write.io.req.bits.data, main_write.io.req.bits.data)
    bank.io.write.io.req.bits.mask := Mux(exec_write_sel, exec_write.io.req.bits.mask, main_write.io.req.bits.mask)
    bank.io.write.is_acc        := Mux(exec_write_sel, exec_isacc_sel, false.B) 
    bank.io.write.bank_id      := Mux(exec_write_sel, exec_bankid, main_bankid)
    bank.io.write.rob_id       := Mux(exec_write_sel, exec_robid, main_robid)

    // Write request ready reverse connection
    main_write.io.req.ready := !exec_write_sel && bank.io.write.io.req.ready
    exec_write.io.req.ready := exec_write_sel && bank.io.write.io.req.ready

    // Tie-off read metadata (unused for now)
    bank.io.read.rob_id  := 0.U
    bank.io.read.is_acc  := false.B
    bank.io.read.bank_id := i.U
  }
}
