// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2022 Jiuyang Liu <liu@jiuyang.me>

package framework.gpdomain.sequencer.decoder.attribute

import framework.gpdomain.sequencer.decoder.T1DecodePattern

object isVector {

  def apply(t1DecodePattern: T1DecodePattern): isVector =
    Seq(
      y _  -> Y,
      n _  -> N,
      dc _ -> DC
    ).collectFirst {
      case (fn, tri) if fn(t1DecodePattern) => isVector(tri)
    }.get

  def y(t1DecodePattern: T1DecodePattern): Boolean = {
    val allMatched = t1DecodePattern.param.allInstructions.filter(i => i.instructionSet.name == "rv_v")
    allMatched.contains(t1DecodePattern.instruction)
  }

  def n(t1DecodePattern: T1DecodePattern): Boolean = {
    val allMatched = t1DecodePattern.param.allInstructions.filter(i => !(y(t1DecodePattern) || dc(t1DecodePattern)))
    allMatched.contains(t1DecodePattern.instruction)
  }

  def dc(t1DecodePattern: T1DecodePattern): Boolean = false
}

case class isVector(value: TriState) extends BooleanDecodeAttribute {
  override val description: String = "This instruction should be decode by T1."
}
