package framework.acc.gpdomain.loader.configs

case class Params(
  iBufWords:  Int,
  constBytes: Int) {
  require(iBufWords > 0)
  require(constBytes > 0)
}
