#ifndef MXINT5GEMV_OPS_H
#define MXINT5GEMV_OPS_H
#include "Acc/BankInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mxint5gemvDialect.h.inc"
#define GET_OP_CLASSES
#include "mxint5gemvOps.h.inc"
#endif
