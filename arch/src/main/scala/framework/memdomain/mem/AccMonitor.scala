package framework.memdomain.mem

import chisel3._
import chisel3.util._

import framework.builtin.util.Util._

import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.balldomain.blink.{SramReadWithInfo, SramWriteWithInfo}

class AccMonitorWriteIO(n: Int, w: Int, mask_len: Int)(implicit b: CustomBuckyballConfig, p: Parameters)extends Bundle {
  val io = new SramWriteIO(n, w, mask_len)

  val rob_id = Input(UInt(log2Up(b.rob_entries).W))
  val is_acc = Input(Bool())
  val bank_id = Input(UInt(log2Up(b.sp_banks+b.acc_banks).W))
}

class AccMonitorReadIO(n: Int, w: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle{
  val io = new SramReadIO(n, w)
  
  val rob_id = Input(UInt(log2Up(b.rob_entries).W))
  val is_acc = Input(Bool())
  val bank_id = Input(UInt(log2Up(b.sp_banks+b.acc_banks).W))
}

class AccPipe(val n: Int, val w: Int, val mask_len: Int) (implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val write_in  = new AccMonitorWriteIO(n, w, mask_len)           // outer —> Acc
    val read      = Flipped(new AccMonitorReadIO(n, w))            // Acc <—> SramBank
    val write_out = Flipped(new SramWriteIO(n, w, mask_len)) // Acc —> SramBank
  })

  // Pipeline registers
  val valid_reg = RegInit(false.B)
  val addr_reg  = RegInit(0.U(log2Ceil(n).W))
  val data_reg  = RegInit(0.U(w.W))
  val mask_reg  = RegInit(VecInit(Seq.fill(mask_len)(false.B)))

  // Default/forward metadata (so it's always driven)
  io.read.rob_id  := io.write_in.rob_id
  io.read.is_acc  := io.write_in.is_acc
  io.read.bank_id := io.write_in.bank_id


  when (io.write_in.is_acc || RegNext(io.write_in.is_acc)) {
// -----------------------------------------------------------------------------
// exec->AccPipe->SramBank
// -----------------------------------------------------------------------------
    // Stage 1: Read request
    io.read.io.req.valid        := io.write_in.io.req.valid
    io.read.io.req.bits.addr    := io.write_in.io.req.bits.addr
    // AccPipe read is not from DMA
    io.read.io.req.bits.fromDMA := false.B
    valid_reg                := io.write_in.io.req.valid
    addr_reg                 := io.write_in.io.req.bits.addr
    data_reg                 := io.write_in.io.req.bits.data
    mask_reg                 := io.write_in.io.req.bits.mask

    // Stage 2: Accumulate (when read data is ready)
    val acc_data = WireDefault(0.U(w.W))
    when (valid_reg && io.read.io.resp.valid) {
      acc_data := data_reg + io.read.io.resp.bits.data
    }.otherwise {
      acc_data := data_reg
    }

    // Stage 3: Write back
    io.write_out.req.valid     := valid_reg && io.read.io.resp.valid
    io.write_out.req.bits.addr := addr_reg
    io.write_out.req.bits.data := acc_data
    io.write_out.req.bits.mask := mask_reg

    // Backpressure
    io.write_in.io.req.ready      := io.read.io.req.ready
    io.read.io.resp.ready         := io.write_out.req.ready
  }.otherwise {
// -----------------------------------------------------------------------------
// main->SramBank
// -----------------------------------------------------------------------------
    io.read.io.req.valid          := false.B
    io.read.io.req.bits.addr      := 0.U(log2Ceil(n).W)
    io.read.io.req.bits.fromDMA   := false.B

    io.write_out.req.valid     := io.write_in.io.req.valid
    io.write_out.req.bits.addr := io.write_in.io.req.bits.addr
    io.write_out.req.bits.data := io.write_in.io.req.bits.data
    io.write_out.req.bits.mask := io.write_in.io.req.bits.mask

    io.write_in.io.req.ready      := io.write_out.req.ready
    io.read.io.resp.ready         := false.B
  }
}


class AccReadRouter(val n: Int, val w: Int)(implicit b: CustomBuckyballConfig, p: Parameters)  extends Module {
  val io = IO(new Bundle {
    val read_in1 = new AccMonitorReadIO(n, w)
    val read_in2 = new AccMonitorReadIO(n, w)
    val read_out = Flipped(new SramReadIO(n, w))
  })

// -----------------------------------------------------------------------------
// Arbiter - use two Arbiters to handle req and resp separately
// -----------------------------------------------------------------------------
  // Priority arbiter, read_in2 has index 0 for higher priority
  val req_arbiter = Module(new Arbiter(new SramReadReq(n), 2))
  req_arbiter.io.in(0) <> io.read_in2.io.req
  req_arbiter.io.in(1) <> io.read_in1.io.req
  io.read_out.req <> req_arbiter.io.out

  // Response distributor: record which input initiated the request
  val resp_to_in1 = RegNext(req_arbiter.io.chosen === 1.U && req_arbiter.io.out.fire, false.B)
  val resp_to_in2 = RegNext(req_arbiter.io.chosen === 0.U && req_arbiter.io.out.fire, false.B)

  // Response distribution
  io.read_in1.io.resp.valid := io.read_out.resp.valid && resp_to_in1
  io.read_in1.io.resp.bits  := io.read_out.resp.bits
  io.read_in2.io.resp.valid := io.read_out.resp.valid && resp_to_in2
  io.read_in2.io.resp.bits  := io.read_out.resp.bits

  // Response ready signal
  io.read_out.resp.ready :=
    (resp_to_in1 && io.read_in1.io.resp.ready) ||
    (resp_to_in2 && io.read_in2.io.resp.ready)

  assert(!(io.read_in1.io.req.valid && io.read_in2.io.req.valid), "[AccBank Router]: Read requests is not allowed at the same time")
}


class AccMonitor(n: Int, w: Int, aligned_to: Int, single_ported: Boolean) (implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val mask_len = (w / (aligned_to * 8)) max 1
  val mask_elem = UInt((w min (aligned_to * 8)).W)

  val io = IO(new Bundle {
    val read  = new AccMonitorReadIO(n, w)
    val write = new AccMonitorWriteIO(n, w, mask_len)
  })

  val sram = Module(new SramBank(n, w, aligned_to, single_ported))
  val pipe = Module(new AccPipe(n, w, mask_len))
  val read_router = Module(new AccReadRouter(n, w))

// -----------------------------------------------------------------------------
// Write request enters pipeline
// -----------------------------------------------------------------------------
  pipe.io.write_in <> io.write

// -----------------------------------------------------------------------------
// Read request arbitration
// -----------------------------------------------------------------------------
  read_router.io.read_in1 <> pipe.io.read
  read_router.io.read_in2 <> io.read

  // Connect AccRouter output to SramBank
  sram.io.read <> read_router.io.read_out

// -----------------------------------------------------------------------------
// Pipeline output connected to underlying SRAM write port
// -----------------------------------------------------------------------------
  sram.io.write <> pipe.io.write_out

}
