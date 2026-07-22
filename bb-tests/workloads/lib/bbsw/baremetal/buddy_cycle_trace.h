#ifndef BUDDY_CYCLE_TRACE_H
#define BUDDY_CYCLE_TRACE_H

#include <stdint.h>

void _mlir_ciface_buddyTraceCycleStart(int64_t id);
void _mlir_ciface_buddyTraceCycleEnd(int64_t id);
void _mlir_ciface_buddyTraceCycleStartPath(int64_t id, int64_t depth,
                                           int64_t path0, int64_t path1,
                                           int64_t path2, int64_t path3);
void _mlir_ciface_buddyTraceCycleEndPath(int64_t id, int64_t depth,
                                         int64_t path0, int64_t path1,
                                         int64_t path2, int64_t path3);

#endif
