package framework.builtin.mem

import chisel3._
import chisel3.util._

import framework.builtin.util.Util._

class AccWriteIO(n: Int, w: Int, mask_len: Int) extends SramWriteIO(n, w, mask_len) {
  val is_acc = Input(Bool())
}

class AccPipe(val n: Int, val w: Int, val mask_len: Int) extends Module {
  val io = IO(new Bundle {
    val write_in  = new AccWriteIO(n, w, mask_len)           // outer —> Acc
    val read      = Flipped(new SramReadIO(n, w))            // Acc <—> SramBank
    val write_out = Flipped(new SramWriteIO(n, w, mask_len)) // Acc —> SramBank 
  })

  // Pipeline registers
  val valid_reg = RegInit(false.B)
  val addr_reg  = RegInit(0.U(log2Ceil(n).W))
  val data_reg  = RegInit(0.U(w.W))
  val mask_reg  = RegInit(VecInit(Seq.fill(mask_len)(false.B)))
  
  when (io.write_in.is_acc || RegNext(io.write_in.is_acc)) {
// -----------------------------------------------------------------------------
// exec->AccPipe->SramBank
// -----------------------------------------------------------------------------
    // Stage 1: Read request
    io.read.req.valid        := io.write_in.req.valid
    io.read.req.bits.addr    := io.write_in.req.bits.addr
    io.read.req.bits.fromDMA := false.B  // AccPipe读取不是来自DMA
    valid_reg                := io.write_in.req.valid
    addr_reg                 := io.write_in.req.bits.addr
    data_reg                 := io.write_in.req.bits.data
    mask_reg                 := io.write_in.req.bits.mask
    
    // Stage 2: Accumulate (when read data is ready)
    val acc_data = WireDefault(0.U(w.W))
    when (valid_reg && io.read.resp.valid) {
      acc_data := data_reg + io.read.resp.bits.data
    }.otherwise {
      acc_data := data_reg
    }
    
    // Stage 3: Write back
    io.write_out.req.valid     := valid_reg && io.read.resp.valid
    io.write_out.req.bits.addr := addr_reg
    io.write_out.req.bits.data := acc_data
    io.write_out.req.bits.mask := mask_reg
    
    // Backpressure
    io.write_in.req.ready      := io.read.req.ready
    io.read.resp.ready         := io.write_out.req.ready
  }.otherwise {
// -----------------------------------------------------------------------------
// main->SramBank
// -----------------------------------------------------------------------------
    io.read.req.valid          := false.B
    io.read.req.bits.addr      := 0.U(log2Ceil(n).W)
    io.read.req.bits.fromDMA   := false.B

    io.write_out.req.valid     := io.write_in.req.valid
    io.write_out.req.bits.addr := io.write_in.req.bits.addr
    io.write_out.req.bits.data := io.write_in.req.bits.data
    io.write_out.req.bits.mask := io.write_in.req.bits.mask
    
    io.write_in.req.ready      := io.write_out.req.ready
    io.read.resp.ready         := false.B
  }
}


class AccReadRouter(val n: Int, val w: Int) extends Module {
  val io = IO(new Bundle {
    val read_in1 = new SramReadIO(n, w)
    val read_in2 = new SramReadIO(n, w)
    val read_out = Flipped(new SramReadIO(n, w))
  })

// -----------------------------------------------------------------------------
// 仲裁器 - 使用两个Arbiter分别处理req和resp
// -----------------------------------------------------------------------------
  //  priority arbiter, read_in2 has index 0 for higher priority
  val req_arbiter = Module(new Arbiter(new SramReadReq(n), 2))
  req_arbiter.io.in(0) <> io.read_in2.req
  req_arbiter.io.in(1) <> io.read_in1.req
  io.read_out.req <> req_arbiter.io.out
  
  // 响应分发器：记录哪个输入发起了请求
  val resp_to_in1 = RegNext(req_arbiter.io.chosen === 1.U && req_arbiter.io.out.fire, false.B)
  val resp_to_in2 = RegNext(req_arbiter.io.chosen === 0.U && req_arbiter.io.out.fire, false.B)
  
  // 响应分发
  io.read_in1.resp.valid := io.read_out.resp.valid && resp_to_in1
  io.read_in1.resp.bits  := io.read_out.resp.bits
  io.read_in2.resp.valid := io.read_out.resp.valid && resp_to_in2
  io.read_in2.resp.bits  := io.read_out.resp.bits
  
  // 响应的ready信号
  io.read_out.resp.ready := 
    (resp_to_in1 && io.read_in1.resp.ready) ||
    (resp_to_in2 && io.read_in2.resp.ready)

  assert(!(io.read_in1.req.valid && io.read_in2.req.valid), "[AccBank Router]: Read requests is not allowed at the same time")
}


class AccBank(n: Int, w: Int, aligned_to: Int, single_ported: Boolean) extends Module {
  val mask_len = (w / (aligned_to * 8)) max 1
  val mask_elem = UInt((w min (aligned_to * 8)).W)

  val io = IO(new Bundle {
    val read  = new SramReadIO(n, w)
    val write = new AccWriteIO(n, w, mask_len)
  })

  val sram = Module(new SramBank(n, w, aligned_to, single_ported))
  val pipe = Module(new AccPipe(n, w, mask_len))
  val read_router = Module(new AccReadRouter(n, w))

// -----------------------------------------------------------------------------
// 写请求进流水线
// -----------------------------------------------------------------------------
  pipe.io.write_in <> io.write

// -----------------------------------------------------------------------------
// 读请求仲裁
// -----------------------------------------------------------------------------
  read_router.io.read_in1 <> pipe.io.read
  read_router.io.read_in2 <> io.read

  // 连接AccRouter的输出到SramBank
  sram.io.read <> read_router.io.read_out

// -----------------------------------------------------------------------------
// 流水线输出连到底层SRAM写端
// -----------------------------------------------------------------------------
  sram.io.write <> pipe.io.write_out

}
