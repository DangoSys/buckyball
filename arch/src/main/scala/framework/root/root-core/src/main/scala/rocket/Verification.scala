package hier.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import freechips.rocketchip.tile.TraceBundle
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.coherence.Coherence
import memcore.memory.coherence.configs.CoherenceParams
import memcore.bus.chi.Fabric
import memcore.bus.chi.rnf.RnfParams
import memcore.bus.chi.snf.{LineRequest, LineResponse}

/** Verification system: the production Core's data and instruction L1s share a canonical Home. */
@instantiable
class Verification(
  memory:                 CoherenceParams,
  regions:                Seq[PhysicalRegion],
  topName:                String,
  commands:               Option[Commands] = None
)(
  implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  override def desiredName: String = topName
  require(memory.agents == 2)
  private val cp = CpuMemParams(memory.chi, tagBits = 6)

  @public
  val io = IO(new Bundle {
    val timerInterrupt             = Input(Bool())
    val softwareInterrupt          = Input(Bool())
    val externalInterrupt          = Input(Bool())
    val resetVector                = Input(UInt(64.W))
    val time                       = Input(UInt(64.W))
    val memoryRequest              = Decoupled(new LineRequest(memory.chi))
    val memoryResponse             = Flipped(Decoupled(new LineResponse(memory.chi)))
    val uncachedRequest            = Decoupled(new UncachedRequest(cp))
    val uncachedResponse           = Flipped(Decoupled(new UncachedResponse(cp)))
    val trace                      = Output(new TraceBundle)
    val retired                    = Output(Bool())
    val retiredPc                  = Output(UInt(64.W))
    val trapped                    = Output(Bool())
    val trapCause                  = Output(UInt(64.W))
    val trapValue                  = Output(UInt(64.W))
    val trapPc                     = Output(UInt(64.W))
    val cancelledData              = Output(Bool())
    val memoryOutstanding          = Output(UInt(log2Ceil(memory.mshrEntries + 1).W))
    val blockMaintenanceResponse   = commands.map(_ => Input(Bool()))
    val maintenanceResponsePending = commands.map(_ => Output(Bool()))
    val admission                  = commands.map(mode => new AdmissionPorts(mode.tracking, cpuParams.core.nPMPs, memory.chi))
  })

  val config = RnfParams(memory.chi, nodeId = 1, cacheLines = 8, homeId = memory.homeId, homeCount = 1, banks = 2)
  val core:   Instance[Core]      = Instantiate(new Core(config, config.copy(nodeId = 2), regions, commands))
  val home:   Instance[Coherence] = Instantiate(new Coherence(memory))
  val fabric: Instance[Fabric]    = Instantiate(new Fabric(memory.chi, memory.agents, memory.homeId))
  commands.foreach(_ => io.admission.get <> core.io.admission.get)
  core.io.resetVector                 := io.resetVector
  core.io.time                        := io.time
  core.io.timerInterrupt              := io.timerInterrupt
  core.io.softwareInterrupt           := io.softwareInterrupt
  core.io.externalInterrupt           := io.externalInterrupt
  core.io.supervisorExternalInterrupt := false.B
  core.io.hartId                      := 0.U
  io.uncachedRequest <> core.io.uncachedRequest
  core.io.uncachedResponse <> io.uncachedResponse
  io.trace                            := core.io.trace
  io.retired                          := core.io.retired
  io.retiredPc                        := core.io.retiredPc
  io.trapped                          := core.io.trapped
  io.trapCause                        := core.io.trapCause
  io.trapValue                        := core.io.trapValue
  io.trapPc                           := core.io.trapPc
  io.cancelledData                    := core.io.cancelledData
  fabric.io.active                    := true.B
  fabric.io.blockRequesterRsp         := false.B
  fabric.io.requesters(0) <> core.io.chi
  commands.foreach { _ =>
    val response      = fabric.io.requesters(0).rxRsp
    val isMaintenance = response.bits.txnId === config.banks.U
    val hold          = io.blockMaintenanceResponse.get && isMaintenance
    core.io.chi.rxRsp.valid           := response.valid && !hold
    response.ready                    := core.io.chi.rxRsp.ready && !hold
    io.maintenanceResponsePending.get := response.valid && isMaintenance
  }
  fabric.io.requesters(1) <> core.io.instructionChi
  home.io.req <> fabric.io.req
  home.io.rxRsp <> fabric.io.rxRsp
  home.io.rxDat <> fabric.io.rxDat
  fabric.io.rsp <> home.io.rsp
  fabric.io.dat <> home.io.dat
  fabric.io.snp <> home.io.snp
  io.memoryRequest <> home.io.memoryReq
  home.io.memoryResp <> io.memoryResponse
  io.memoryOutstanding                := home.io.outstanding
}
