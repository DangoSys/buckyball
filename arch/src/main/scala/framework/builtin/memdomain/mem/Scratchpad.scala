package framework.builtin.memdomain.mem

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._

import framework.builtin.util.Util._
import examples.BuckyballConfigs.CustomBuckyballConfig

import framework.builtin.memdomain.mem.AccWriteIO


class Scratchpad(config: CustomBuckyballConfig)(implicit val p: Parameters) extends Module with HasCoreParameters {
  import config._

  // Assertion: ensure configuration consistency
  assert(sp_singleported, "Scratchpad expects single-ported SRAM banks")

  val io = IO(new Bundle {
    // SRAM read/write interface - used by load/store
    val dma = new Bundle {
      val sramread  = Vec(sp_banks, new SramReadIO(spad_bank_entries, spad_w))
      val sramwrite = Vec(sp_banks, new SramWriteIO(spad_bank_entries, spad_w, spad_mask_len))
      val accread   = Vec(acc_banks, new SramReadIO(acc_bank_entries, acc_w))
      val accwrite  = Vec(acc_banks, new SramWriteIO(acc_bank_entries, acc_w, acc_mask_len))
    }
    // Execution unit read/write interface - one read and write per bank, OpA and OpB guaranteed to access different banks
    val exec = new Bundle {
      val sramread  = Vec(sp_banks, new SramReadIO(spad_bank_entries, spad_w))
      val sramwrite = Vec(sp_banks, new SramWriteIO(spad_bank_entries, spad_w, spad_mask_len))
      val accread   = Vec(acc_banks, new SramReadIO(acc_bank_entries, acc_w))
      val accwrite  = Vec(acc_banks, new SramWriteIO(acc_bank_entries, acc_w, acc_mask_len))
    }
  })

// -----------------------------------------------------------------------------
// Scratchpad
// -----------------------------------------------------------------------------

  // SRAM banks - each bank has only one port, supports simultaneous read and write
  val spad_mems = Seq.fill(sp_banks) { Module(new SramBank(
    spad_bank_entries, spad_w,
    // Use configuration parameters
    aligned_to, sp_singleported
  )) }

  // Request arbitration and connection for each bank
  spad_mems.zipWithIndex.foreach { case (bank, i) =>

    // All read request sources
    val main_read_req = io.dma.sramread(i).req
    val exec_read_req = io.exec.sramread(i).req

    // All write request sources
    val main_write = io.dma.sramwrite(i)
    val exec_write = io.exec.sramwrite(i)

    // Assertion: OpA and OpB should not access the same bank simultaneously
    assert(!(exec_read_req.valid && exec_write.req.valid),
      s"Bank ${i}: exec and write cannot access the same bank simultaneously")

    // Read request arbitration: priority exec > main
    val exec_read_sel = exec_read_req.valid
    val main_read_sel = main_read_req.valid && !exec_read_sel

    // Write request arbitration: exec has higher priority
    val exec_write_sel = exec_write.req.valid

    // Connect read request to SramBank
    bank.io.read.req.valid := exec_read_sel || main_read_sel
    bank.io.read.req.bits := Mux(exec_read_sel, exec_read_req.bits, main_read_req.bits)

    // Read request ready reverse connection
    main_read_req.ready := main_read_sel && bank.io.read.req.ready
    exec_read_req.ready := exec_read_sel && bank.io.read.req.ready

    // Record which client initiated read request (for response distribution)
    val resp_to_main = RegNext(main_read_sel && bank.io.read.req.fire, false.B)
    val resp_to_exec = RegNext(exec_read_sel && bank.io.read.req.fire, false.B)

    // Read response distribution
    io.dma.sramread(i).resp.valid := bank.io.read.resp.valid && resp_to_main
    io.dma.sramread(i).resp.bits := bank.io.read.resp.bits

    io.exec.sramread(i).resp.valid := bank.io.read.resp.valid && resp_to_exec
    io.exec.sramread(i).resp.bits := bank.io.read.resp.bits

    // Read response ready signal: either client ready is sufficient
    bank.io.read.resp.ready :=
      (resp_to_main && io.dma.sramread(i).resp.ready) ||
      (resp_to_exec && io.exec.sramread(i).resp.ready)

    // Connect write request to SramBank
    bank.io.write.req.valid     := Mux(exec_write_sel, exec_write.req.valid, main_write.req.valid)
    bank.io.write.req.bits.addr := Mux(exec_write_sel, exec_write.req.bits.addr, main_write.req.bits.addr)
    bank.io.write.req.bits.data := Mux(exec_write_sel, exec_write.req.bits.data, main_write.req.bits.data)
    bank.io.write.req.bits.mask := Mux(exec_write_sel, exec_write.req.bits.mask, main_write.req.bits.mask)

    // Write request ready reverse connection
    main_write.req.ready := !exec_write_sel && bank.io.write.req.ready
    exec_write.req.ready := exec_write_sel && bank.io.write.req.ready
  }

// -----------------------------------------------------------------------------
// Accumulator
// -----------------------------------------------------------------------------

  val acc_mems = Seq.fill(acc_banks) { Module(new AccBank(
    acc_bank_entries, acc_w,
    // Use configuration parameters
    aligned_to, sp_singleported
  )) }

  // Request arbitration and connection for each acc bank
  acc_mems.zipWithIndex.foreach { case (bank, i) =>
  // All read request sources
    val main_read_req = io.dma.accread(i).req
    val exec_read_req = io.exec.accread(i).req

    // All write request sources
    val main_write = io.dma.accwrite(i)
    val exec_write = io.exec.accwrite(i)

    // Assertion: OpA and OpB should not access the same bank simultaneously
    assert(!(exec_read_req.valid && exec_write.req.valid),
      s"Bank ${i}: exec and write cannot access the same bank simultaneously")

    // Read request arbitration: priority exec > main
    val exec_read_sel = exec_read_req.valid
    val main_read_sel = main_read_req.valid && !exec_read_sel

    // Write request arbitration: exec has higher priority
    val exec_write_sel = exec_write.req.valid

    // Connect read request to SramBank
    bank.io.read.req.valid := exec_read_sel || main_read_sel
    bank.io.read.req.bits := Mux(exec_read_sel, exec_read_req.bits, main_read_req.bits)

    // Read request ready reverse connection
    main_read_req.ready := main_read_sel && bank.io.read.req.ready
    exec_read_req.ready := exec_read_sel && bank.io.read.req.ready

    // Record which client initiated read request (for response distribution)
    val resp_to_main = RegNext(main_read_sel && bank.io.read.req.fire, false.B)
    val resp_to_exec = RegNext(exec_read_sel && bank.io.read.req.fire, false.B)

    // Read response distribution
    io.dma.accread(i).resp.valid := bank.io.read.resp.valid && resp_to_main
    io.dma.accread(i).resp.bits := bank.io.read.resp.bits

    io.exec.accread(i).resp.valid := bank.io.read.resp.valid && resp_to_exec
    io.exec.accread(i).resp.bits := bank.io.read.resp.bits

    // Read response ready signal: either client ready is sufficient
    bank.io.read.resp.ready :=
      (resp_to_main && io.dma.accread(i).resp.ready) ||
      (resp_to_exec && io.exec.accread(i).resp.ready)

    // Connect write request to SramBank
    bank.io.write.req.valid       := Mux(exec_write_sel, exec_write.req.valid,     main_write.req.valid)
    bank.io.write.req.bits.addr   := Mux(exec_write_sel, exec_write.req.bits.addr, main_write.req.bits.addr)
    bank.io.write.req.bits.data   := Mux(exec_write_sel, exec_write.req.bits.data, main_write.req.bits.data)
    bank.io.write.req.bits.mask   := Mux(exec_write_sel, exec_write.req.bits.mask, main_write.req.bits.mask)
    bank.io.write.is_acc          := Mux(exec_write_sel, true.B, false.B)

    // Write request ready reverse connection
    main_write.req.ready := !exec_write_sel && bank.io.write.req.ready
    exec_write.req.ready := exec_write_sel && bank.io.write.req.ready
  }
}
