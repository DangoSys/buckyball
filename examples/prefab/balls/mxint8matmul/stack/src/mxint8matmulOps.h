//===- mxint8matmulOps.h - Mxint8matmulBall operations ----------------===//

#ifndef MXINT8MATMUL_OPS_H
#define MXINT8MATMUL_OPS_H

#include "Acc/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxint8matmulDialect.h.inc"

#define GET_OP_CLASSES
#include "mxint8matmulOps.h.inc"

#endif // MXINT8MATMUL_OPS_H
