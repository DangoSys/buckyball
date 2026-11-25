package framework.generic.contract.ball

import framework.generic.contract.blink.BlinkParam

/**
 * What the Ball domain (BBus, channels, MMIO bank, SubROB) knows of the accelerator that hosts it:
 * the link widths and the few host properties its container is sized by.
 */
case class BallHostParam(
  link:             BlinkParam,
  robEntries:       Int,
  bankRequestPorts: Int,
  mmioEnable:       Boolean,
  mmioBankNum:      Int,
  mmioBankEntries:  Int,
  mmioBankWidth:    Int,
  subRobEnable:     Boolean,
  subRobDepth:      Int,
  bankAliasEnable:  Boolean,
  // The host has a GP domain that calls Balls as operators.
  operators:        Boolean,
  // The width of the DDR addresses an expanding Ball writes into the instructions it expands.
  memAddrBits:      Int)
