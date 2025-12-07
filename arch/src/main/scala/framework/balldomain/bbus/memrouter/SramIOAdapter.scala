package framework.balldomain.bbus.memrouter


import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.rs.{BallRsIssue, BallRsComplete}
import framework.memdomain.mem.{SramReadIO, SramWriteIO, SramReadReq, SramReadResp, SramWriteReq}
import framework.balldomain.blink.{SramReadWithInfo, SramWriteWithInfo}

class SramIOAdapter(numBalls: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
    val io = IO(new Bundle {
        val sramWrite_i =  new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len)
        val sramRead_o =  new SramReadWithInfo(b.spad_bank_entries, b.spad_w)
    })

    val read_o = Wire(new SramReadWithInfo(b.spad_bank_entries, b.spad_w))

    read_o.io.req.ready := true.B
    io.sramWrite_i.io.req.ready := read_o.io.resp.ready
    read_o.io.resp.valid := io.sramWrite_i.io.req.valid
    read_o.io.resp.bits.data := io.sramWrite_i.io.req.bits.data
    read_o.io.resp.bits.fromDMA := false.B

    read_o.rob_id := io.sramWrite_i.rob_id
    read_o.is_acc := io.sramWrite_i.is_acc
    read_o.bank_id := io.sramWrite_i.bank_id  

    io.sramRead_o <> read_o  
}
