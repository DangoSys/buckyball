#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Device-visible heap for compiled programs and their jobs. */
void workspace_init(void *memory, size_t bytes);
void workspace_begin(void *memory, size_t bytes);
void *workspace_alloc(size_t bytes);
void workspace_free(void *memory);
size_t workspace_peak(void);

#ifdef __cplusplus
}
#endif
