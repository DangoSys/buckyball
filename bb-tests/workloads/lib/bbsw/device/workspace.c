#include "workspace.h"
#include <stdio.h>
#include <stdlib.h>

/* One device-visible heap per core worker; blocks are first fit and coalesce on
 * free. */

struct block {
  size_t size;
  struct block *next, *previous;
  uint64_t available;
};

static _Thread_local struct block *head;
static _Thread_local unsigned char *end;
static _Thread_local size_t used, peak;

void workspace_init(void *memory, size_t bytes) {
  if ((uintptr_t)memory % 16 || bytes < sizeof(struct block))
    abort();
  // Populate user mappings before the heap is used.
  for (size_t offset = 0; offset < bytes; offset += 4096)
    ((volatile unsigned char *)memory)[offset] = 0;
}

/* Starts an empty heap for the calling core worker. */
void workspace_begin(void *memory, size_t bytes) {
  head = memory;
  *head = (struct block){.size = bytes - sizeof(struct block), .available = 1};
  used = peak = 0;
  end = (unsigned char *)memory + bytes;
}

size_t workspace_peak(void) {
  if (!head)
    abort();
  return peak;
}

void *workspace_alloc(size_t bytes) {
  if (!head || bytes > SIZE_MAX - 15)
    abort();
  bytes = (bytes + 15) & ~(size_t)15;
  for (struct block *block = head; block; block = block->next) {
    if (!block->available || block->size < bytes)
      continue;
    if (block->size - bytes >= sizeof(struct block) + 16) {
      struct block *next =
          (struct block *)((unsigned char *)(block + 1) + bytes);
      *next = (struct block){.size = block->size - bytes - sizeof(*next),
                             .next = block->next,
                             .previous = block,
                             .available = 1};
      if (next->next)
        next->next->previous = next;
      block->next = next;
      block->size = bytes;
    }
    block->available = 0;
    used += block->size + sizeof(*block);
    if (used > peak)
      peak = used;
    return block + 1;
  }
  fputs("workspace exhausted\n", stderr);
  abort();
}

void workspace_free(void *memory) {
  if (!memory)
    return;
  struct block *block = (struct block *)memory - 1;
  if ((uintptr_t)block < (uintptr_t)head ||
      (uintptr_t)memory >= (uintptr_t)end || block->available) {
    fputs("invalid workspace free\n", stderr);
    abort();
  }
  block->available = 1;
  used -= block->size + sizeof(*block);
  if (block->next && block->next->available) {
    struct block *next = block->next;
    block->size += sizeof(*block) + next->size;
    block->next = next->next;
    if (block->next)
      block->next->previous = block;
  }
  if (block->previous && block->previous->available) {
    struct block *previous = block->previous;
    previous->size += sizeof(*block) + block->size;
    previous->next = block->next;
    if (previous->next)
      previous->next->previous = previous;
  }
}
