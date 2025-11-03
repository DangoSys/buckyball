#ifndef _DMA_H_
#define _DMA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DMA_BANDWIDTH 128

// Get DMA row width (bytes)
static inline uint32_t dma_row_bytes() { return DMA_BANDWIDTH / 8; }

#ifdef __cplusplus
}
#endif

#endif // _DMA_H_
