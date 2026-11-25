package framework.generic.acc.gpdomain.loader.configs

import framework.generic.acc.gpdomain.loader.configs.Params

case class Params(
  iBufWords:  Int,
  constBytes: Int) {
  require(iBufWords > 0)
  require(constBytes > 0)
}
