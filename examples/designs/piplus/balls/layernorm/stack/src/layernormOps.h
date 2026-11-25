//===- layernormOps.h - LayerNormBall operations
//------------------------------===//

#ifndef LAYERNORM_OPS_H
#define LAYERNORM_OPS_H

#include "Acc/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "layernormDialect.h.inc"

#define GET_OP_CLASSES
#include "layernormOps.h.inc"

#endif // LAYERNORM_OPS_H
