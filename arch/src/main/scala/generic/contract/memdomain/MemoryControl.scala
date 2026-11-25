package framework.generic.contract.memdomain

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}

import framework.generic.contract.memdomain.toddr.DmaError
import framework.generic.contract.memdomain.DmaFaultControl
import framework.generic.contract.memdomain.DmaPageFault

import framework.generic.contract.memdomain.MemoryContext

import framework.generic.memory.interlock.{Params => TrackingParam}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.memory.access.preflight.PmpEntry
import framework.generic.contract.memdomain.toddr.DmaStatus

/** Address-space state frozen with the instruction and consumed only by memory requests. */
class MemoryContext(permission: PmpParams) extends Bundle {
  val satp      = UInt(64.W)
  val privilege = UInt(2.W)
  val sum       = Bool()
  val mxr       = Bool()
  val pmp       =
    Vec(permission.entries, new PmpEntry(permission))
}

/**
 * The DMA page fault of an accelerator instruction: its virtual address, whether it is a store,
 * and the satp the instruction was issued under.
 */
class DmaPageFault extends Bundle {
  val write   = Bool()
  val address = UInt(64.W)
  val satp    = UInt(64.W)
}

/**
 * Every page an instruction's DMA touches is translated before any of its data moves, so a page
 * fault stalls the instruction precisely, with no partial effect. The hart that issued it sees
 * the fault and retries the translation (resolve true) or fails the instruction (false).
 */
class DmaFaultControl extends Bundle {
  val fault   = Valid(new DmaPageFault)
  val resolve = Flipped(Valid(Bool()))
}
