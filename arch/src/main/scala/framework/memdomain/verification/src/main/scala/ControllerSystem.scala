package framework.memdomain.verification

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import framework.top.GlobalConfig
import framework.system.core.ControllerAdmission
import framework.system.core.rocket.RoCCIO
import framework.system.tile.BankNetwork
import hier.core.rocket.AdmissionPorts
import hier.tile.TaskController
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.bus.chi.{Params => ChiParams}
import memcore.memory.bank.{Bank, BankRequest, BankSetParams}

/** Real controller/task/bank-move composition; CPU command and bank initialization are test peers. */
@instantiable
class ControllerSystem(b: GlobalConfig, signatures: Seq[BigInt])(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  private val tracking   = TrackingParams()
  private val bankParams =
    BankSetParams(b.memDomain.bankWidth, 1, b.memDomain.bankEntries, math.max(1, log2Ceil(b.frontend.rob_entries)))
  private val cores      = b.memDomain.computeCoreIds

  @public
  val io = IO(new Bundle {
    val core    = Flipped(new AdmissionPorts(tracking, nPMPs, ChiParams()))
    val workers = Vec(signatures.size, new RoCCIO(64))

    val access = Vec(
      cores.size,
      Flipped(new memcore.memory.mesh_shm.MeshLocalBankPort(
        16,
        10,
        bankParams.dataBits,
        bankParams.tagBits
      ))
    )

    val allowBankResponse = Input(Vec(cores.size, Bool()))
    val moveFault         = Valid(new framework.memdomain.frontend.mem.dma.DmaStatus)
    val moveAccepted      = Output(Bool())
    val moveCompleted     = Output(Bool())
    val bankWriteAccepted = Output(Vec(cores.size, Bool()))
  })

  val admission = Instantiate(new ControllerAdmission(tracking, ChiParams(), moves = true))
  val tasks     = Instantiate(new TaskController(1 to signatures.size, signatures, b.memDomain.nCores))
  val network   = Instantiate(new BankNetwork(b, cores, useMesh = true, controllerMove = true))
  admission.io.core <> io.core
  admission.io.task <> tasks.io.ports(0)
  tasks.io.satp := admission.io.taskSatp
  for (i <- signatures.indices) { tasks.io.ports(i + 1) <> io.workers(i) }
  admission.io.move <> network.io.controllerMvover.get
  io.moveFault     := admission.io.moveFault
  io.moveAccepted  := admission.io.move.command.fire
  io.moveCompleted := admission.io.move.completion.fire
  network.io.hartIds.zipWithIndex.foreach { case (hart, i) => hart := i.U }
  for (i <- cores.indices) {
    val port = network.io.compute(i)
    port.config.valid          := false.B; port.config.bits       := 0.U.asTypeOf(port.config.bits)
    port.queryValid            := false.B; port.queryVbank        := 0.U; port.barrierArrive := false.B
    port.move.command.valid    := false.B; port.move.command.bits := 0.U.asTypeOf(port.move.command.bits)
    port.move.completion.ready := true.B
    for (q <- port.requests) {
      q.read.req.valid  := false.B; q.read.req.bits  := 0.U.asTypeOf(q.read.req.bits); q.read.resp.ready   := true.B
      q.write.req.valid := false.B; q.write.req.bits := 0.U.asTypeOf(q.write.req.bits); q.write.resp.ready := true.B
      q.bank_id         := 0.U; q.group_id           := 0.U; q.is_shared                                   := false.B
      q.hart_id         := 0.U; q.rob_id             := 0.U; q.inst_id                                     := 0.U
    }
    // Bank zero is the declared private endpoint used by this gate. Storage and ACK are the existing Bank IP.
    val bank = Instantiate(new Bank(bankParams))
    val choose = Module(new Arbiter(chiselTypeOf(port.local.request.bits), 2))
    choose.io.in(0) <> port.local.request
    choose.io.in(1) <> io.access(i).request
    bank.io.request.valid      := choose.io.out.valid
    choose.io.out.ready        := bank.io.request.ready
    bank.io.request.bits       := 0.U.asTypeOf(new BankRequest(bankParams))
    bank.io.request.bits.addr  := choose.io.out.bits.addr
    bank.io.request.bits.write := choose.io.out.bits.write
    bank.io.request.bits.data  := choose.io.out.bits.data
    bank.io.request.bits.mask  := choose.io.out.bits.mask
    bank.io.request.bits.tag   := choose.io.out.bits.tag
    val owner = Reg(UInt(1.W))
    io.bankWriteAccepted(i)     := choose.io.in(0).fire && choose.io.in(0).bits.write
    when(choose.io.out.fire) {
      assert(
        choose.io.out.bits.bank === 0.U && choose.io.out.bits.addr < bankParams.entriesPerBank.U,
        "ControllerSystem endpoint is private bank zero within the actual bank capacity"
      )
      owner := choose.io.chosen
    }
    port.local.response.bits    := bank.io.response.bits
    io.access(i).response.bits  := bank.io.response.bits
    port.local.response.valid   := bank.io.response.valid && owner === 0.U && io.allowBankResponse(i)
    io.access(i).response.valid := bank.io.response.valid && owner === 1.U
    bank.io.response.ready      := Mux(
      owner === 0.U,
      port.local.response.ready && io.allowBankResponse(i),
      io.access(i).response.ready
    )
  }
}
