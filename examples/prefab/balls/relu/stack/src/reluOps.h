//===- reluOps.h - ReluBall operations ------------------------------------===//

#ifndef RELU_OPS_H
#define RELU_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "reluDialect.h.inc"

#define GET_OP_CLASSES
#include "reluOps.h.inc"

#endif // RELU_OPS_H
