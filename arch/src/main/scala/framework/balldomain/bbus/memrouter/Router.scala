package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/**
 * Router:
 *
 * Input: valid signals for each input channel and channel num requests
 * Output: source input channel ID for each output channel
 */
@instantiable
class Router(val numInputs: Int, val numOutputs: Int, val maxPerChannelWidth: Int) extends Module {
  val inputChannelIdWidth  = log2Ceil(numInputs)
  val outputChannelIdWidth = log2Ceil(numOutputs)

  @public
  val io = IO(new Bundle {
    val in  = Flipped(Valid(Vec(numInputs, UInt(maxPerChannelWidth.W))))
    val out = Vec(numOutputs, Decoupled(UInt(inputChannelIdWidth.W)))
  })

}
