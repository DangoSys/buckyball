#ifndef FIXED_OPS_H
#define FIXED_OPS_H
#include "Acc/BankInterfaces.h"
#include "fixedDialect.h.inc"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "fixedOps.h.inc"
#endif
