package framework.bbus.memrouter

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.memdomain.mem.{SramReadIO, SramWriteIO, SramReadReq, SramReadResp, SramWriteReq}
import framework.blink.{SramReadWithInfo, SramWriteWithInfo}
import framework.bbus.BBusConfigIO

class MemRouter(numBalls: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  private val numBanks = b.sp_banks + b.acc_banks
  val io = IO(new Bundle {
    val sramRead_i = Vec(numBalls, Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w)))
    val sramWrite_i = Vec(numBalls, Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))

    val bbusConfig_i = Flipped(Decoupled(new BBusConfigIO(numBalls)))

    val sramRead_o = Vec(numBanks, Flipped(new SramReadWithInfo(b.spad_bank_entries, b.spad_w)))
    val sramWrite_o = Vec(numBanks, Flipped(new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))

  })

  val list_valid = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val list_dst_bid = RegInit(VecInit(Seq.fill(numBalls)(0.U(log2Ceil(numBalls).W))))

  // Explicitly initialize to false
  val memReq = WireInit(VecInit(Seq.fill(numBalls)(false.B)))


  // Default assignment
  io.sramRead_o := DontCare
  io.sramWrite_o := DontCare

  for (i <- 0 until numBalls) {
    io.sramRead_i(i).foreach(_.io.req.ready := false.B)
    io.sramRead_i(i).foreach(_.io.resp.valid := false.B)
    io.sramRead_i(i).foreach(_.io.resp.bits := DontCare)
    io.sramWrite_i(i).foreach(_.io.req.ready := false.B)
  }

  // Routing selection
  for (i <- 0 until numBalls) {
/*
    memReq(i) := io.sramRead_i(i).map(_.io.req.valid).reduce(_||_) ||
                 io.sramWrite_i(i).map(_.io.req.valid).reduce(_||_) ||
                 io.accRead_i(i).map(_.io.req.valid).reduce(_||_)   ||
                 io.accWrite_i(i).map(_.io.req.valid).reduce(_||_)

    when (memReq(i)) {
      io.sramRead_o <> io.sramRead_i(i).io
      io.sramWrite_o <> io.sramWrite_i(i).io
      io.accRead_o <> io.accRead_i(i).io
      io.accWrite_o <> io.accWrite_i(i).io
    }
    */

    for(j <- 0 until numBanks){
        when(io.sramRead_i(i)(j).io.req.valid){
            io.sramRead_o(j).io.req <> io.sramRead_i(i)(j).io.req
        }
    }
    for(j <- 0 until numBanks){
        when(io.sramRead_o(j).io.resp.valid){
          io.sramRead_i(i)(j).io.resp <> io.sramRead_o(j).io.resp
        }
    }

    for(j <- 0 until numBanks){
        when(io.sramWrite_i(i)(j).io.req.valid){
            io.sramWrite_o(j)<> io.sramWrite_i(i)(j)
        }
    }
  }
  io.bbusConfig_i.ready := true.B
  when (io.bbusConfig_i.valid) {
    when(io.bbusConfig_i.bits.set === 1.U){
      list_valid(io.bbusConfig_i.bits.src_bid) := true.B
      list_dst_bid(io.bbusConfig_i.bits.src_bid) := io.bbusConfig_i.bits.dst_bid
    }.otherwise{
      list_valid(io.bbusConfig_i.bits.src_bid) := false.B
      list_dst_bid(io.bbusConfig_i.bits.src_bid) := 0.U
    }
  }

  for(i <- 0 until numBalls){
    when(list_valid(i)){
      val sramIOadapter = Module(new SramIOAdapter(numBalls)(b, p))
      val dst_bid = list_dst_bid(i)
      sramIOadapter.io.sramWrite_i <> io.sramWrite_i(i)(0)
      io.sramRead_i(dst_bid)(0) <> sramIOadapter.io.sramRead_o
      io.sramWrite_o(0).io.req.valid := false.B
      io.sramWrite_o(0).io.req.bits := DontCare
    }
  }
  override lazy val desiredName = "MemRouter"
}
