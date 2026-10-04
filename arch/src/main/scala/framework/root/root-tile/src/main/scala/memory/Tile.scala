package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import hier.core.rocket.Core
import memcore.bus.chi._
import memcore.bus.chi.rnf.RnfParams
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.coherence.configs.CoherenceParams

@instantiable
class Tile(memory: CoherenceParams, l1: RnfParams, regions: Seq[PhysicalRegion])(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  require(l1.chi == memory.chi && l1.homeId == memory.homeId && l1.homeCount == 1)
  private val c  = memory.chi
  private val cp = CpuMemParams(c, tagBits = 6)

  private val cores = memory.agents / 2
  require(memory.agents == 2 * cores)

  @public
  val io = IO(new Bundle {
    val resetVector      = Input(UInt(64.W))
    val memoryReq        = Decoupled(new LineRequest(c))
    val memoryResp       = Flipped(Decoupled(new LineResponse(c)))
    val uncachedRequest  = Vec(cores, Decoupled(new UncachedRequest(cp)))
    val uncachedResponse = Vec(cores, Flipped(Decoupled(new UncachedResponse(cp))))
    val retired          = Output(Vec(cores, Bool()))
    val retiredPc        = Output(Vec(cores, UInt(64.W)))
    val trapped          = Output(Vec(cores, Bool()))
    val trapCause        = Output(Vec(cores, UInt(64.W)))
    val trapValue        = Output(Vec(cores, UInt(64.W)))
    val trapPc           = Output(Vec(cores, UInt(64.W)))
    val outstanding      = Output(UInt(log2Ceil(memory.mshrEntries + 1).W))
    val observedReq      = Output(Valid(new RequestFlit(c)))
    val observedRsp      = Output(Valid(new ResponseFlit(c)))
    val observedDat      = Output(Valid(new DataFlit(c)))
    val observedSnp      = Output(Valid(new DirectedSnoop(c)))
    val observedRxRsp    = Output(Valid(new ResponseFlit(c)))
    val observedRxDat    = Output(Valid(new DataFlit(c)))
  })

  val home = Instantiate(new Home(memory))
  home.io.active            := true.B
  home.io.blockRequesterRsp := false.B
  io.memoryReq <> home.io.memoryReq
  home.io.memoryResp <> io.memoryResp
  io.outstanding            := home.io.outstanding
  io.observedReq            := home.io.observedReq
  io.observedRsp            := home.io.observedRsp
  io.observedDat            := home.io.observedDat
  io.observedSnp            := home.io.observedSnp
  io.observedRxRsp          := home.io.observedRxRsp
  io.observedRxDat          := home.io.observedRxDat
  // Requester i is core i's data L1; requester cores + i is its instruction L1.
  val core =
    Seq.tabulate(cores)(i => Instantiate(new Core(l1.copy(nodeId = i + 1), l1.copy(nodeId = cores + i + 1), regions)))
  for (i <- 0 until cores) {
    core(i).io.timerInterrupt              := false.B
    core(i).io.softwareInterrupt           := false.B
    core(i).io.externalInterrupt           := false.B
    core(i).io.supervisorExternalInterrupt := false.B
    core(i).io.hartId                      := i.U
    core(i).io.resetVector                 := io.resetVector
    home.io.requesters(i) <> core(i).io.chi
    home.io.requesters(cores + i) <> core(i).io.instructionChi
    io.uncachedRequest(i) <> core(i).io.uncachedRequest
    core(i).io.uncachedResponse <> io.uncachedResponse(i)
    io.retired(i)                          := core(i).io.retired
    io.retiredPc(i)                        := core(i).io.retiredPc
    io.trapped(i)                          := core(i).io.trapped
    io.trapCause(i)                        := core(i).io.trapCause
    io.trapValue(i)                        := core(i).io.trapValue
    io.trapPc(i)                           := core(i).io.trapPc
  }
}
