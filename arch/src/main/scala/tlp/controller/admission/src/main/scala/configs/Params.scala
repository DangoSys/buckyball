package framework.tlp.controller.admission.configs

import upickle.default.{macroRW, ReadWriter}
import framework.generic.contract.memory.access.preflight.configs.PmpParams

case class Params(permission: PmpParams, moves: Boolean)
object Params { implicit val rw: ReadWriter[Params] = macroRW }
