package framework.generic.contract.tlink

import chisel3._
import chisel3.util._
import framework.generic.contract.memory.access.preflight.PmpEntry
import framework.generic.contract.memory.access.preflight.configs.PmpParams

class Start(taskBits: Int) extends Bundle {

  val task     = UInt(taskBits.W)
  val entry    = UInt(64.W)
  val codeEnd  = UInt(64.W)
  val argument = UInt(64.W)
  val stack    = UInt(64.W)
}

class Completion(taskBits: Int) extends Bundle {

  val task      = UInt(taskBits.W)
  val value     = UInt(64.W)
  val cancelled = Bool()
}

/** Address space and permissions captured from the task's launch command. */
class TaskContext(permission: PmpParams) extends Bundle {
  val satp               = UInt(64.W)
  val effectivePrivilege = UInt(2.W)
  val sum                = Bool()
  val mxr                = Bool()
  val pmp                =
    Vec(permission.entries, new PmpEntry(permission))
}

/** One accepted launch binds the task context and starts its local program together. */
class Launch(taskBits: Int, permission: PmpParams) extends Bundle {
  val descriptor = new Start(taskBits)
  val context    = new TaskContext(permission)
}

/** Byte offsets into the TSS and the selected private store; transfers use 8-byte beats. */
class Transfer extends Bundle {
  val code         = Bool()
  val toLocal      = Bool()
  val localOffset  = UInt(32.W)
  val sharedOffset = UInt(32.W)
  val bytes        = UInt(32.W)
}

/** One thread task: launch, cancellation and completion. Storage accesses use independent interfaces. */
class Task(taskBits: Int, permission: PmpParams) extends Bundle {
  val transfer    = Flipped(Decoupled(new Transfer))
  val transferred = Decoupled(Bool())
  val inUse       = Input(Bool())
  val start       = Flipped(Decoupled(new Launch(taskBits, permission)))
  val result      = Decoupled(new Completion(taskBits))
  val cancel      = Input(Bool())
  val running     = Output(Bool())
  val storageBusy = Output(Bool())

}
