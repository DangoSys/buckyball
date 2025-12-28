package framework.balldomain.prototype.vector.bond

import chisel3._
import chisel3.util._

abstract class BondWrapper {
  val bondName = "vvv"

  def to[T](name: String)(body: => T): T =
    body

  def from[T](name: String)(body: => T): T =
    body
}
