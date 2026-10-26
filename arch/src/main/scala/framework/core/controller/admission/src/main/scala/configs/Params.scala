package framework.core.controller.admission.configs

import framework.core.memory.access.preflight.configs.PmpParams
import upickle.default.{macroRW, ReadWriter}

case class Params(permission: PmpParams, moves: Boolean)
object Params { implicit val rw: ReadWriter[Params] = macroRW }
