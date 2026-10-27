//===- smatmulOps.h - SMatMulBall operations ------------------------------===//

#ifndef SMATMUL_OPS_H
#define SMATMUL_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "smatmulDialect.h.inc"

#define GET_OP_CLASSES
#include "smatmulOps.h.inc"

#endif // SMATMUL_OPS_H
