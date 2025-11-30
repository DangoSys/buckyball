package framework.bbus.memrouter


import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.memdomain.mem.{SramReadIO, SramWriteIO, SramReadReq, SramReadResp, SramWriteReq}

class SramIOAdapter(numBalls: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
    val io = IO(new Bundle {
        val sramWrite_i =  new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)
        val sramRead_o =  new SramReadIO(b.spad_bank_entries, b.spad_w)
    })
    io.sramRead_o.req.ready := true.B
    io.sramWrite_i.req.ready := io.sramRead_o.resp.ready
    io.sramRead_o.resp.valid := io.sramWrite_i.req.valid
    io.sramRead_o.resp.bits.data := io.sramWrite_i.req.bits.data
    io.sramRead_o.resp.bits.fromDMA := false.B
}
