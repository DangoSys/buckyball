package framework.top.configs

import org.chipsalliance.cde.config.{Config, Field}
import upickle.default._

case class SimParam(diffTest: Boolean = false, accessTest: Boolean = false) {
  require(!(diffTest && accessTest))
}

object SimParam {
  implicit val rw: ReadWriter[SimParam] = macroRW
}

case object SimParamKey extends Field[SimParam](SimParam())

class WithSimParam(param: SimParam)
    extends Config((site, here, up) => {
      case SimParamKey => param
    })
