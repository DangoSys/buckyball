package framework.core.execution.privileged.configs

import framework.core.memory.access.preflight.configs.PmpParams
import upickle.default.{macroRW, ReadWriter}

case class Params(permission: PmpParams)
object Params { implicit val rw: ReadWriter[Params] = macroRW }
