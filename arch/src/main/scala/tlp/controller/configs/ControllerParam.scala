package framework.tlp.controller.configs

import framework.generic.contract.tlink.TLinkParam

import framework.tlp.tss.configs.{Params => TssParam}
import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.generic.contract.memory.access.preflight.configs.PmpParams

case class CommandsParam(
  tracking:    TrackingParam,
  dataBits:    Int,
  permission:  PmpParams,
  moves:       Boolean,
  taskEnabled: Boolean,
  remote:      Boolean)

case class TasksParam(
  locals:     Seq[TLinkParam],
  signatures: Seq[BigInt],
  tss:        TssParam,
  permission: PmpParams) {
  val count = signatures.size
  require(count > 0 && locals.size == count && tss.ports == count)
  require(locals.forall(p => p.tss == tss.memory && p.taskBits <= 64))
}

case class ControllerParam(commands: CommandsParam, tasks: Option[TasksParam])
