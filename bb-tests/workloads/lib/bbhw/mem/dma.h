#ifndef _DMA_H_
#define _DMA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DMA_BANDWIDTH 128

// 获取DMA的行宽度（字节数）
static inline uint32_t dma_row_bytes() { return DMA_BANDWIDTH / 8; }

#ifdef __cplusplus
}
#endif

#endif // _DMA_H_
