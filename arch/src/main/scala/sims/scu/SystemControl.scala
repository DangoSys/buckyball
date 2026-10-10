package sims.scu

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.util._
import memcore.memory.cpu.{CpuMemParams, UncachedRequest, UncachedResponse}

/**
 * SCU on the System device ports, with the register layout documented on SCUParams. The hart sub-region is
 * selected by address, so software keeps using `base + hartId * stride` regardless of port.
 * Data is right-aligned at `addr`, as on every uncached CPU port.
 */
@instantiable
class SystemControl(ports: Int, p: CpuMemParams, params: SCUParams = SCUParams()) extends Module {
  val strideBits = log2Ceil(params.strideBytes)

  @public
  val io = IO(new Bundle {
    val request     = Vec(ports, Flipped(Decoupled(new UncachedRequest(p))))
    val response    = Vec(ports, Decoupled(new UncachedResponse(p)))
    val hartIds     = Input(Vec(ports, UInt(32.W)))
    val failure     = Input(Vec(ports, Valid(new framework.memdomain.frontend.mem.dma.DmaStatus)))
    val workDrained = Input(Vec(ports, Bool()))
  })

  val faulted = SystemControl.observe(io.failure.toSeq, io.hartIds.toSeq)

  for (i <- 0 until ports) {
    val request  = io.request(i)
    val response = io.response(i)
    val write: Instance[SCUWriteDPI] = Instantiate(new SCUWriteDPI)
    val read:  Instance[SCUReadDPI]  = Instantiate(new SCUReadDPI)
    val ready = RegInit(0.U(8.W))

    val pending = RegInit(false.B)
    val tag     = Reg(UInt(p.tagBits.W))
    val error   = Reg(Bool())
    val data    = Reg(UInt(64.W))
    val status  = Reg(Bool())

    val offset   = request.bits.addr - params.baseAddress.U
    val hart     = offset(31, strideBits)
    val register = offset(strideBits - 1, 0)
    val inside   = request.bits.addr >= params.baseAddress.U &&
      request.bits.addr < (params.baseAddress + params.totalSizeBytes).U && hart < params.maxHarts.U
    val fire     = request.fire
    val wr       = request.bits.write
    val exit     = inside && register === 0x00000.U && wr
    val uartTx   = inside && register === 0x20000.U && wr
    val uartRx   = inside && register === 0x20004.U && !wr
    val uartSt   = inside && register === 0x20005.U && !wr
    val readyReg = inside && register === 0x20006.U

    write.io.clock        := clock
    write.io.reset        := reset.asBool
    write.io.exit_hart_id := hart
    write.io.exit_valid   := fire && exit && !faulted
    write.io.exit_code    := request.bits.data(31, 0)
    write.io.uart_hart_id := hart
    write.io.uart_valid   := fire && uartTx
    write.io.uart_data    := request.bits.data(7, 0)

    // The DPI samples on this edge; status reads answer with the fresh sample next cycle.
    read.io.clock   := clock
    read.io.reset   := reset.asBool
    read.io.hart_id := hart
    read.io.enable  := fire && (uartRx || uartSt)
    read.io.pop     := fire && uartRx

    request.ready := !pending && (!request.valid || !exit || io.workDrained(i))
    when(fire) {
      pending                    := true.B
      tag                        := request.bits.tag
      error                      := !(exit || uartTx || uartRx || uartSt || readyReg)
      status                     := uartSt
      data                       := Mux(uartRx, read.io.rx_data, Mux(readyReg && !wr, ready, 0.U))
      when(readyReg && wr)(ready := request.bits.data(7, 0))
    }

    response.valid              := pending
    response.bits.tag           := tag
    response.bits.error         := error
    response.bits.data          := Mux(status, "h60".U | read.io.rx_valid.asUInt, data)
    when(response.fire)(pending := false.B)
  }
}

object SystemControl {

  def observe(failures: Seq[ValidIO[framework.memdomain.frontend.mem.dma.DmaStatus]], hartIds: Seq[UInt]): Bool = {
    val failed  = RegInit(false.B)
    val pending = VecInit(failures.map(_.valid))
    val first   = PriorityEncoder(pending)
    val exit    = Instantiate(new SCUWriteDPI)
    exit.io.clock                   := Module.clock
    exit.io.reset                   := Module.reset.asBool
    exit.io.uart_hart_id            := 0.U
    exit.io.uart_valid              := false.B
    exit.io.uart_data               := 0.U
    exit.io.exit_hart_id            := VecInit(hartIds)(first)
    exit.io.exit_valid              := pending.asUInt.orR && !failed
    exit.io.exit_code               := 0x200.U | VecInit(failures.map(_.bits.error))(first)
    when(pending.asUInt.orR)(failed := true.B)
    for ((failure, index) <- failures.zipWithIndex) {
      when(failure.valid && !failed && first === index.U) {
        printf(
          cf"[SCU] hart ${hartIds(index)} NPU failure: error=${failure.bits.error} address=0x${failure.bits.address}%x\n"
        )
      }
    }
    pending.asUInt.orR || failed
  }

}
