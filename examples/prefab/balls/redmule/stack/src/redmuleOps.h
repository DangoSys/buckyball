//===- redmuleOps.h - RedmuleBall operations ------------------------------===//

#ifndef REDMULE_OPS_H
#define REDMULE_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "redmuleDialect.h.inc"

#define GET_OP_CLASSES
#include "redmuleOps.h.inc"

#endif // REDMULE_OPS_H
