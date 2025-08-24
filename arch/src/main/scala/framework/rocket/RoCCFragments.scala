package framework.rocket

import chisel3._

import org.chipsalliance.cde.config.{Field, Parameters, Config}
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy._
import framework.rocket.{LazyRoCCBB, BuildRoCCBB}

import chipyard.{TestSuitesKey, TestSuiteHelper}

/**
 * Map from a tileId to a particular RoCC accelerator
 */
case object MultiRoCCKey extends Field[Map[Int, Seq[Parameters => LazyRoCC]]](Map.empty[Int, Seq[Parameters => LazyRoCC]])
case object MultiRoCCKeyBB extends Field[Map[Int, Seq[Parameters => LazyRoCCBB]]](Map.empty[Int, Seq[Parameters => LazyRoCCBB]])

/**
 * Config fragment to enable different RoCCs based on the tileId
 */
class WithMultiRoCC extends Config((site, here, up) => {
  case BuildRoCC => site(MultiRoCCKey).getOrElse(site(TileKey).tileId, Nil)
})

class WithMultiRoCCBB extends Config((site, here, up) => {
  case BuildRoCCBB => site(MultiRoCCKeyBB).getOrElse(site(TileKey).tileId, Nil)
})


/**
 * Assigns what was previously in the BuildRoCC key to specific harts with MultiRoCCKey
 * Must be paired with WithMultiRoCC
 */
class WithMultiRoCCFromBuildRoCC(harts: Int*) extends Config((site, here, up) => {
  case BuildRoCC => Nil
  case MultiRoCCKey => up(MultiRoCCKey, site) ++ harts.distinct.map { i =>
    (i -> up(BuildRoCC, site))
  }
})
