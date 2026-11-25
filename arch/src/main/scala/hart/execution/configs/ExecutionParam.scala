package framework.hart.execution.configs

import framework.hart.execution.instruction.configs.{Params => InstructionParam}
import framework.hart.execution.register_file.configs.{Params => RegisterFileParam}
import framework.hart.execution.integer.configs.{Params => IntegerParam}
import framework.hart.execution.muldiv.configs.{Params => MulDivParam}
import framework.hart.execution.fpu.configs.{Params => FpuParam}
import framework.hart.execution.privileged.configs.{Params => PrivilegedParam}
import framework.hart.memory.access.configs.{Params => AccessParam}
import framework.generic.memory.interlock.{Params => TrackingParam}

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
