package framework.generic.contract.ant

import framework.generic.contract.ant.Retire

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.arith.{DivRem, Multiply}
import framework.generic.memory.spm
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.RoCCIO

class Retire extends Bundle {
  val pc          = UInt(64.W)
  val instruction = UInt(32.W)
  val rd          = UInt(5.W)
  val data        = UInt(64.W)
}
