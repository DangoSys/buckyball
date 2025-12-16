// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2022 Jiuyang Liu <liu@jiuyang.me>

package framework.gpdomain.sequencer.decoder.attribute

import framework.gpdomain.sequencer.decoder.T1DecodePattern

object isDivider {
  def apply(t1DecodePattern: T1DecodePattern): isDivider =
    Seq(
      y _  -> Y,
      n _  -> N,
      dc _ -> DC
    ).collectFirst {
      case (fn, tri) if fn(t1DecodePattern) => isDivider(tri)
    }.get

  def y(t1DecodePattern: T1DecodePattern): Boolean = {
    val allMatched = Seq(
      "vdiv.vv",
      "vdiv.vx",
      "vdivu.vv",
      "vdivu.vx",
      "vfdiv.vf",
      "vfdiv.vv",
      "vfrdiv.vf",
      "vfsqrt.v",
      "vrem.vv",
      "vrem.vx",
      "vremu.vv",
      "vremu.vx"
    )
    allMatched.contains(t1DecodePattern.instruction.name)
  }
  def n(t1DecodePattern: T1DecodePattern): Boolean = {
    val allMatched = t1DecodePattern.param.allInstructions.filter(i => !(y(t1DecodePattern) || dc(t1DecodePattern)))
    allMatched.contains(t1DecodePattern.instruction)
  }

  def dc(t1DecodePattern: T1DecodePattern): Boolean = false
}

case class isDivider(value: TriState) extends BooleanDecodeAttribute {
  override val description: String =
    "goes to [[org.chipsalliance.t1.rtl.LaneDiv]] or [[org.chipsalliance.t1.rtl.LaneDivFP]]. if FP exist, all div goes to [[org.chipsalliance.t1.rtl.LaneDivFP]]"
}
