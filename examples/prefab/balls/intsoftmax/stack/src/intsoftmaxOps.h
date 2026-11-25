#ifndef INTSOFTMAX_OPS_H
#define INTSOFTMAX_OPS_H
#include "Acc/BankInterfaces.h"
#include "intsoftmaxDialect.h.inc"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "intsoftmaxOps.h.inc"
#endif
