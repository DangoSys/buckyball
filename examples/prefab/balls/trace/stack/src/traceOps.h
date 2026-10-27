//===- traceOps.h - TraceBall operations ----------------------------------===//

#ifndef TRACE_OPS_H
#define TRACE_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "traceDialect.h.inc"

#define GET_OP_CLASSES
#include "traceOps.h.inc"

#endif // TRACE_OPS_H
