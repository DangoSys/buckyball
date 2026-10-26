package framework.core.execution.configs

import framework.core.execution.instruction.configs.{Params => InstructionParam}
import framework.core.execution.register_file.configs.{Params => RegisterFileParam}
import framework.core.execution.integer.configs.{Params => IntegerParam}
import framework.core.execution.muldiv.configs.{Params => MulDivParam}
import framework.core.execution.fpu.configs.{Params => FpuParam}
import framework.core.execution.privileged.configs.{Params => PrivilegedParam}
import framework.core.memory.access.configs.{Params => AccessParam}
import framework.ip.memory.interlock.{Params => TrackingParam}

case class ExecutionParam(
  hartId:            Int,
  resetVector:       BigInt,
  commandsEnabled:   Boolean,
  directAccelerator: Boolean,
  moves:             Boolean,
  tracking:          TrackingParam,
  mtimeAddress:      BigInt,
  memoryAccess:      AccessParam,
  instruction:       InstructionParam,
  registerFile:      RegisterFileParam,
  integer:           IntegerParam,
  muldiv:            MulDivParam,
  fpu:               FpuParam,
  privileged:        PrivilegedParam)
