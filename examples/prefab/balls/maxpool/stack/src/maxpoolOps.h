//===- maxpoolOps.h - MaxPoolBall operations ------------------------------===//

#ifndef MAXPOOL_OPS_H
#define MAXPOOL_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "maxpoolDialect.h.inc"

#define GET_OP_CLASSES
#include "maxpoolOps.h.inc"

#endif // MAXPOOL_OPS_H
