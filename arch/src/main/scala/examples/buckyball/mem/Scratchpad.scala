package buckyball.mem

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._

import buckyball.util.Util._
import buckyball.BuckyBallConfig

import buckyball.mem.AccWriteIO


class Scratchpad(config: BuckyBallConfig)(implicit val p: Parameters) extends Module with HasCoreParameters {
  import config._

  // 断言：确保配置一致性
  assert(sp_singleported, "Scratchpad expects single-ported SRAM banks")

  val io = IO(new Bundle {
    // SRAM读写接口 - load/store使用
    val dma = new Bundle {
      val sramread  = Vec(sp_banks, new SramReadIO(spad_bank_entries, spad_w))
      val sramwrite = Vec(sp_banks, new SramWriteIO(spad_bank_entries, spad_w, spad_mask_len))
      val accread   = Vec(acc_banks, new SramReadIO(acc_bank_entries, acc_w))
      val accwrite  = Vec(acc_banks, new SramWriteIO(acc_bank_entries, acc_w, acc_mask_len))
    }
    // 执行单元读写接口 - 每个bank一个read和write，OpA和OpB保证访问不同bank
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

  // SRAM banks - 每个bank只有一个端口，支持同时读写
  val spad_mems = Seq.fill(sp_banks) { Module(new SramBank(
    spad_bank_entries, spad_w,
    aligned_to, sp_singleported // 使用配置参数
  )) }

  // 为每个bank进行请求仲裁和连接
  spad_mems.zipWithIndex.foreach { case (bank, i) =>
    
    // 所有的读请求源
    val main_read_req = io.dma.sramread(i).req
    val exec_read_req = io.exec.sramread(i).req

    // 所有的写请求源
    val main_write = io.dma.sramwrite(i)
    val exec_write = io.exec.sramwrite(i)
    
    // 断言：OpA和OpB不应同时访问同一个bank
    assert(!(exec_read_req.valid && exec_write.req.valid), 
      s"Bank ${i}: exec and write cannot access the same bank simultaneously")
    
    // 读请求仲裁：优先级 exec > main
    val exec_read_sel = exec_read_req.valid
    val main_read_sel = main_read_req.valid && !exec_read_sel
    
    // 写请求仲裁：exec有更高优先级
    val exec_write_sel = exec_write.req.valid
    
    // 连接读请求到SramBank
    bank.io.read.req.valid := exec_read_sel || main_read_sel
    bank.io.read.req.bits := Mux(exec_read_sel, exec_read_req.bits, main_read_req.bits)
    
    // 读请求的ready反向连接
    main_read_req.ready := main_read_sel && bank.io.read.req.ready
    exec_read_req.ready := exec_read_sel && bank.io.read.req.ready
    
    // 记录哪个客户端发起了读请求（用于响应分发）
    val resp_to_main = RegNext(main_read_sel && bank.io.read.req.fire, false.B)
    val resp_to_exec = RegNext(exec_read_sel && bank.io.read.req.fire, false.B)
    
    // 读响应分发
    io.dma.sramread(i).resp.valid := bank.io.read.resp.valid && resp_to_main
    io.dma.sramread(i).resp.bits := bank.io.read.resp.bits
    
    io.exec.sramread(i).resp.valid := bank.io.read.resp.valid && resp_to_exec
    io.exec.sramread(i).resp.bits := bank.io.read.resp.bits
    
    // 读响应的ready信号：任一客户端ready即可
    bank.io.read.resp.ready := 
      (resp_to_main && io.dma.sramread(i).resp.ready) ||
      (resp_to_exec && io.exec.sramread(i).resp.ready)
    
    // 连接写请求到SramBank
    bank.io.write.req.valid     := Mux(exec_write_sel, exec_write.req.valid, main_write.req.valid)
    bank.io.write.req.bits.addr := Mux(exec_write_sel, exec_write.req.bits.addr, main_write.req.bits.addr)
    bank.io.write.req.bits.data := Mux(exec_write_sel, exec_write.req.bits.data, main_write.req.bits.data)
    bank.io.write.req.bits.mask := Mux(exec_write_sel, exec_write.req.bits.mask, main_write.req.bits.mask)

    // 写请求的ready反向连接
    main_write.req.ready := !exec_write_sel && bank.io.write.req.ready
    exec_write.req.ready := exec_write_sel && bank.io.write.req.ready
  }

// -----------------------------------------------------------------------------
// Accumulator
// -----------------------------------------------------------------------------

  val acc_mems = Seq.fill(acc_banks) { Module(new AccBank(
    acc_bank_entries, acc_w,
    aligned_to, sp_singleported // 使用配置参数
  )) }

  // 为每个acc bank进行请求仲裁和连接
  acc_mems.zipWithIndex.foreach { case (bank, i) =>
  // 所有的读请求源
    val main_read_req = io.dma.accread(i).req
    val exec_read_req = io.exec.accread(i).req

    // 所有的写请求源
    val main_write = io.dma.accwrite(i)
    val exec_write = io.exec.accwrite(i)

    // 断言：OpA和OpB不应同时访问同一个bank
    assert(!(exec_read_req.valid && exec_write.req.valid), 
      s"Bank ${i}: exec and write cannot access the same bank simultaneously")
    
    // 读请求仲裁：优先级 exec > main
    val exec_read_sel = exec_read_req.valid
    val main_read_sel = main_read_req.valid && !exec_read_sel
    
    // 写请求仲裁：exec有更高优先级
    val exec_write_sel = exec_write.req.valid
    
    // 连接读请求到SramBank
    bank.io.read.req.valid := exec_read_sel || main_read_sel
    bank.io.read.req.bits := Mux(exec_read_sel, exec_read_req.bits, main_read_req.bits)
    
    // 读请求的ready反向连接
    main_read_req.ready := main_read_sel && bank.io.read.req.ready
    exec_read_req.ready := exec_read_sel && bank.io.read.req.ready
    
    // 记录哪个客户端发起了读请求（用于响应分发）
    val resp_to_main = RegNext(main_read_sel && bank.io.read.req.fire, false.B)
    val resp_to_exec = RegNext(exec_read_sel && bank.io.read.req.fire, false.B)
    
    // 读响应分发
    io.dma.accread(i).resp.valid := bank.io.read.resp.valid && resp_to_main
    io.dma.accread(i).resp.bits := bank.io.read.resp.bits

    io.exec.accread(i).resp.valid := bank.io.read.resp.valid && resp_to_exec
    io.exec.accread(i).resp.bits := bank.io.read.resp.bits

    // 读响应的ready信号：任一客户端ready即可
    bank.io.read.resp.ready := 
      (resp_to_main && io.dma.accread(i).resp.ready) ||
      (resp_to_exec && io.exec.accread(i).resp.ready)

    // 连接写请求到SramBank
    bank.io.write.req.valid       := Mux(exec_write_sel, exec_write.req.valid,     main_write.req.valid)
    bank.io.write.req.bits.addr   := Mux(exec_write_sel, exec_write.req.bits.addr, main_write.req.bits.addr)
    bank.io.write.req.bits.data   := Mux(exec_write_sel, exec_write.req.bits.data, main_write.req.bits.data)
    bank.io.write.req.bits.mask   := Mux(exec_write_sel, exec_write.req.bits.mask, main_write.req.bits.mask)
    bank.io.write.is_acc          := Mux(exec_write_sel, true.B, false.B)

    // 写请求的ready反向连接
    main_write.req.ready := !exec_write_sel && bank.io.write.req.ready
    exec_write.req.ready := exec_write_sel && bank.io.write.req.ready
  }
}


