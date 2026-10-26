#ifndef BUCKYBALL_STAGE_TRACE_H
#define BUCKYBALL_STAGE_TRACE_H

#include "mlir/IR/Builders.h"

namespace mlir::buddy {
LogicalResult emitStageTensorTrace(OpBuilder &builder, Location loc,
                                   Value value, DictionaryAttr metadata);
}

#endif
