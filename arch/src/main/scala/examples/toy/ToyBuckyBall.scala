package examples.toy

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile.{HasCoreParameters, LazyRoCC, LazyRoCCModuleImp, OpcodeSet, TileKey}
import freechips.rocketchip.tilelink._

import freechips.rocketchip.tile.{LazyRoCC, LazyRoCCModuleImp}
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.top.GlobalConfig
import examples.toy.balldomain.BallDomain
import framework.frontend.Frontend
import framework.gpdomain.GpDomain
import framework.memdomain.MemDomain

class ToyBuckyball(val b: GlobalConfig)(implicit p: Parameters)
    extends LazyRoCC(opcodes = OpcodeSet.custom3, nPTWPorts = 1) {

  // the width of core's register file
  val xLen = p(TileKey).core.xLen

  val id_node   = TLIdentityNode()
  val xbar_node = TLXbar()

  xbar_node := TLBuffer()                                  := id_node
  xbar_node := TLBuffer()                                  := id_node
  id_node   := TLWidthWidget(b.memDomain.dma_buswidth / 8) := TLBuffer() := xbar_node

  override lazy val module = new ToyBuckyballModule(this)

  override val tlNode  = id_node
  override val atlNode = TLIdentityNode()
  val node             = tlNode
}

class ToyBuckyballModule(outer: ToyBuckyball) extends LazyRoCCModuleImp(outer) with HasCoreParameters {
  import outer.b._
  val edge: TLEdgeOut    = outer.id_node.edges.out.head
  val b:    GlobalConfig = outer.b

  val frontend:   Instance[Frontend]   = Instantiate(new Frontend(b))
  val ballDomain: Instance[BallDomain] = Instantiate(new BallDomain(b))
  val memDomain:  Instance[MemDomain]  = Instantiate(new MemDomain(b)(edge))
  val gpDomain:   Instance[GpDomain]   = Instantiate(new GpDomain(b))

  frontend.io.cmd.valid    := io.cmd.valid
  frontend.io.cmd.bits.cmd := io.cmd.bits
  io.cmd.ready             := frontend.io.cmd.ready

  // Frontend -> BallDomain
  ballDomain.global_issue_i <> frontend.io.ball_issue_o
  frontend.io.ball_complete_i <> ballDomain.global_complete_o

  // Frontend -> MemDomain
  memDomain.io.global_issue_i <> frontend.io.mem_issue_o
  frontend.io.mem_complete_i <> memDomain.io.global_complete_o

  // PTW connected to MemDomain's TLB (shared TLB has only 1 PTW port)
  io.ptw(0) <> memDomain.io.ptw(0)

  // TLB exception handling - shared TLB has only 1 exception interface
  // Set flush input signals
  memDomain.io.tlbExp(0).flush_skip  := false.B
  memDomain.io.tlbExp(0).flush_retry := false.B

  // Frontend -> GpDomain
  gpDomain.io.global_issue_i <> frontend.io.gp_issue_o
  frontend.io.gp_complete_i <> gpDomain.io.global_complete_o

  ballDomain.bankRead <> memDomain.io.ballDomain.bankRead
  ballDomain.bankWrite <> memDomain.io.ballDomain.bankWrite

  io.resp <> frontend.io.resp
  io.busy      := frontend.io.busy
  io.interrupt := memDomain.io.tlbExp(0).interrupt

  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(frontend.io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "ToyBuckyball: busy for too long!")

}
