// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2022 Jiuyang Liu <liu@jiuyang.me>

package framework.gpdomain.sequencer.decoder.attribute

import framework.gpdomain.sequencer.decoder.T1DecodePattern

object isFcompare {
  def apply(t1DecodePattern: T1DecodePattern): isFcompare =
    Seq(
      y _  -> Y,
      n _  -> N,
      dc _ -> DC
    ).collectFirst {
      case (fn, tri) if fn(t1DecodePattern) => isFcompare(tri)
    }.get

  def y(t1DecodePattern: T1DecodePattern): Boolean = {
    val allMatched = Seq(
      "vfmax.vf",
      "vfmax.vv",
      "vfmin.vf",
      "vfmin.vv",
      "vfredmax.vs",
      "vfredmin.vs",
      "vmfeq.vf",
      "vmfeq.vv",
      "vmfge.vf",
      "vmfgt.vf",
      "vmfle.vf",
      "vmfle.vv",
      "vmflt.vf",
      "vmflt.vv",
      "vmfne.vf",
      "vmfne.vv"
    )
    allMatched.contains(t1DecodePattern.instruction.name)
  }
  def n(t1DecodePattern: T1DecodePattern): Boolean = {
    val allMatched = t1DecodePattern.param.allInstructions.filter(i => !(y(t1DecodePattern) || dc(t1DecodePattern)))
    allMatched.contains(t1DecodePattern.instruction)
  }

  def dc(t1DecodePattern: T1DecodePattern): Boolean = false
}

case class isFcompare(value: TriState) extends BooleanDecodeAttribute {
  override val description: String = "TODO: remove it, but remains attribute."
}
