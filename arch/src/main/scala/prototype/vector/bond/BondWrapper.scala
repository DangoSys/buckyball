package prototype.vector.bond

import org.chipsalliance.cde.config._
import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy._
import org.chipsalliance.diplomacy.bundlebridge._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.diplomacy.nodes._

abstract class BondWrapper(implicit p: Parameters) extends LazyModule {
  val bondName = "vvv"

  def to[T](name: String)(body: => T): T = {
    LazyScope(s"bond_to_${name}", s"Bond_${bondName}_to_${name}") { body }
  }

  def from[T](name: String)(body: => T): T = {
    LazyScope(s"bond_from_${name}", s"Bond_${bondName}_from_${name}") { body }
  }
}
