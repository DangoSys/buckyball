// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.

package framework.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.IntParam

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._

import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy.lazymodule._

import freechips.rocketchip.rocket.{
  CanHavePTW,
  CanHavePTWModule,
  HellaCacheIO,
  MStatus,
  M_SZ,
  M_XRD,
  PRV,
  PTE,
  SimpleHellaCacheIF,
  TLBPTWIO
}
import freechips.rocketchip.tilelink.{TLClientNode, TLIdentityNode, TLMasterParameters, TLMasterPortParameters, TLNode}
import freechips.rocketchip.util.InOrderArbiter

case object BuildRoCCBB extends Field[Seq[Parameters => LazyRoCCBB]](Nil)

class RoCCCommandBB(implicit p: Parameters) extends CoreBundle()(p) {
  val inst   = new RoCCInstruction
  val rs1    = Bits(xLen.W)
  val rs2    = Bits(xLen.W)
  val status = new MStatus
}

class RoCCResponseBB(implicit p: Parameters) extends CoreBundle()(p) {
  val rd   = Bits(5.W)
  val data = Bits(xLen.W)
}

class RoCCCoreIOBB(val nRoCCCSRs: Int = 0)(implicit p: Parameters) extends CoreBundle()(p) {
  val cmd       = Flipped(Decoupled(new RoCCCommandBB))
  val resp      = Decoupled(new RoCCResponseBB)
  val mem       = new HellaCacheIO
  val busy      = Output(Bool())
  val interrupt = Output(Bool())
  val exception = Input(Bool())
  val csrs      = Flipped(Vec(nRoCCCSRs, new CustomCSRIO))
}

class RoCCIOBB(val nPTWPorts: Int, nRoCCCSRs: Int)(implicit p: Parameters) extends RoCCCoreIOBB(nRoCCCSRs)(p) {
  val ptw      = Vec(nPTWPorts, new TLBPTWIO)
  val fpu_req  = Decoupled(new FPInput)
  val fpu_resp = Flipped(Decoupled(new FPResult))
}

/** Base classes for Diplomatic TL2 RoCC units * */
abstract class LazyRoCCBB(
  val opcodes:   OpcodeSet,
  val nPTWPorts: Int = 0,
  val usesFPU:   Boolean = false,
  val roccCSRs:  Seq[CustomCSR] = Nil
)(
  implicit p:    Parameters)
    extends LazyModule {
  val module: LazyRoCCModuleImpBB
  require(roccCSRs.map(_.id).toSet.size == roccCSRs.size)
  val atlNode: TLNode = TLIdentityNode()
  val tlNode:  TLNode = TLIdentityNode()
  val stlNode: TLNode = TLIdentityNode()
}

class LazyRoCCModuleImpBB(outer: LazyRoCCBB) extends LazyModuleImp(outer) {
  val io = IO(new RoCCIOBB(outer.nPTWPorts, outer.roccCSRs.size))
  io := DontCare
}

/** Mixins for including RoCC * */

trait HasLazyRoCCBB extends CanHavePTW { this: BaseTile =>
  val roccs    = p(BuildRoCCBB).map(_(p))
  val roccCSRs = roccs.map(_.roccCSRs) // the set of custom CSRs requested by all roccs
  require(
    roccCSRs.flatten.map(_.id).toSet.size == roccCSRs.flatten.size,
    "LazyRoCC instantiations require overlapping CSRs"
  )
  roccs.map(_.atlNode).foreach(atl => tlMasterXbar.node :=* atl)
  roccs.map(_.tlNode).foreach(tl => tlOtherMastersNode :=* tl)
  roccs.map(_.stlNode).foreach(stl => stl :*= tlSlaveXbar.node)

  nPTWPorts += roccs.map(_.nPTWPorts).sum
  nDCachePorts += roccs.size
}

trait HasLazyRoCCModuleBB extends CanHavePTWModule with HasCoreParameters { this: RocketTileModuleImpBB =>

  val (respArb, cmdRouter) = if (outer.roccs.nonEmpty) {
    val respArb      = Module(new RRArbiter(new RoCCResponse()(outer.p), outer.roccs.size))
    // Get usingRVVRoCC from tile parameters
    val usingRVVRoCC = outer.rocketParams.asInstanceOf[RocketTileParamsBB].usingRVVRoCC
    val cmdRouter    = Module(new RoccCommandRouterBB(outer.roccs.map(_.opcodes), usingRVVRoCC)(outer.p))
    outer.roccs.zipWithIndex.foreach {
      case (rocc, i) =>
        rocc.module.io.ptw ++=: ptwPorts
        rocc.module.io.cmd <> cmdRouter.io.out(i)
        val dcIF = Module(new SimpleHellaCacheIF()(outer.p))
        dcIF.io.requestor <> rocc.module.io.mem
        dcachePorts += dcIF.io.cache
        respArb.io.in(i) <> Queue(rocc.module.io.resp)
    }
    (Some(respArb), Some(cmdRouter))
  } else {
    (None, None)
  }

  val roccCSRIOs = outer.roccs.map(_.module.io.csrs)
}

class RoccCommandRouterBB(opcodes: Seq[OpcodeSet], usingRVVRoCC: Boolean)(implicit p: Parameters)
    extends CoreModule()(p) {

  val io = IO(new Bundle {
    val in   = Flipped(Decoupled(new RoCCCommand))
    val out  = Vec(opcodes.size, Decoupled(new RoCCCommand))
    val busy = Output(Bool())
  })

  val cmd = Queue(io.in)

  // Check which opcodes match
  val opcodeMatches = opcodes.map(opcode => opcode.matches(cmd.bits.inst.opcode))
  val anyMatch      = opcodeMatches.reduce(_ || _)

  val cmdReadys = io.out.zip(opcodeMatches).zipWithIndex.map {
    case ((out, matches), i) =>
      // In RVVRoCC mode: route unmatched opcodes to first RoCC (index 0)
      // Otherwise: only route if opcode matches
      val shouldRoute = if (usingRVVRoCC && i == 0) {
        matches || !anyMatch // First RoCC gets matched opcodes OR unmatched opcodes
      } else {
        matches // Other RoCCs only get their matched opcodes
      }

      out.valid := cmd.valid && shouldRoute
      out.bits  := cmd.bits
      out.ready && shouldRoute
  }

  cmd.ready := cmdReadys.reduce(_ || _)
  io.busy   := cmd.valid

  // In RVVRoCC mode, allow unmatched opcodes to go to first RoCC
  // Otherwise, assert that only one RoCC matches
  if (!usingRVVRoCC) {
    assert(PopCount(cmdReadys) <= 1.U, "Custom opcode matched for more than one accelerator")
  }
}
