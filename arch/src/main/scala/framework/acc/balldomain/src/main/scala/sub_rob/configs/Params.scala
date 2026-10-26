package framework.acc.balldomain.sub_rob.configs

case class Params(depth: Int, masterEntries: Int, balls: Int) {
  require(depth > 1)
  require(masterEntries > 1)
  require(balls > 0)
}
