//===- toint8Ops.h - ToInt8Ball operations --------------------------------===//

#ifndef TOINT8_OPS_H
#define TOINT8_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "toint8Dialect.h.inc"

#define GET_OP_CLASSES
#include "toint8Ops.h.inc"

#endif // TOINT8_OPS_H
