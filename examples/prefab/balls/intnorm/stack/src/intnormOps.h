#ifndef INTNORM_OPS_H
#define INTNORM_OPS_H
#include "Acc/BankInterfaces.h"
#include "intnormDialect.h.inc"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "intnormOps.h.inc"
#endif
