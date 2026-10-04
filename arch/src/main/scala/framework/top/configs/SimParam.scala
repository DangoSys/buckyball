package framework.top.configs

import upickle.default._

case class SimParam(diffTest: Boolean = false)

object SimParam {
  implicit val rw: ReadWriter[SimParam] = macroRW
}
