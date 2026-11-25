package framework.generic.contract.controller

import chisel3._
import chisel3.util._
import framework.generic.memory.interlock.{Tag, Params => TrackingParam}
import framework.generic.contract.frontend.{Command, Response}
import framework.generic.contract.memory.access.preflight.PmpEntry
import framework.generic.contract.memory.access.preflight.configs.PmpParams

// The command path between a CPU and its hart's controller (the threads'
// tasks and bank moves): the base CPU speaks it, thread-level parallelism
// serves it.

class CommandSnapshot(tracking: TrackingParam, dataBits: Int, permission: PmpParams) extends Bundle {

  val tag                = UInt(tracking.idBits.W)
  val instruction        = new Command(dataBits)
  val satp               = UInt(64.W)
  val effectivePrivilege = UInt(2.W)
  val sum                = Bool()
  val mxr                = Bool()
  val pmp                = Vec(permission.entries, new PmpEntry(permission))
}

/** Execution-side task and bank-move admission, with completion ordering. */
class Commands(tracking: TrackingParam, dataBits: Int, permission: PmpParams) extends Bundle {
  val command  = Decoupled(new CommandSnapshot(tracking, dataBits, permission))
  val response = Flipped(Decoupled(new Response(dataBits)))
  val complete = Flipped(Decoupled(new Tag(tracking)))
}

/** Decoded management instruction issued by the execution hart. */
class Management extends Bundle {
  val operation = UInt(7.W)
  val context   = UInt(32.W)
  val field     = UInt(32.W)
  val data      = UInt(64.W)
}

class TaskEndpoint extends Bundle {
  val request = Decoupled(new Management)
  val reply   = Flipped(Decoupled(UInt(64.W)))
  val busy    = Input(Bool())
}
