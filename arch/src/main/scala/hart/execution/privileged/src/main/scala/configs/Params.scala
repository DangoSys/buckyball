package framework.hart.execution.privileged.configs

import upickle.default.{macroRW, ReadWriter}
import framework.generic.contract.memory.access.preflight.configs.PmpParams

case class Params(permission: PmpParams)
object Params { implicit val rw: ReadWriter[Params] = macroRW }
