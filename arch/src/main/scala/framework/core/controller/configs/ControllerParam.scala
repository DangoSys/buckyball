package framework.core.controller.configs

import framework.thread.ant.configs.AntParam
import framework.core.tss.configs.{Params => TssParam}
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}

case class CommandsParam(
  tracking:    TrackingParam,
  dataBits:    Int,
  permission:  PmpParams,
  moves:       Boolean,
  taskEnabled: Boolean,
  remote:      Boolean)

case class TasksParam(
  locals:     Seq[AntParam],
  signatures: Seq[BigInt],
  tss:        TssParam,
  permission: PmpParams) {
  val count = signatures.size
  require(count > 0 && locals.size == count && tss.ports == count)
  require(locals.forall(p => p.shared == tss.memory && p.taskBits <= 64))
}

case class ControllerParam(commands: CommandsParam, tasks: Option[TasksParam])
