#ifndef _BANK_H_
#define _BANK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *name_;
  uint32_t base_addr_;
  uint32_t row_num_;
  uint32_t elem_num_;
  uint32_t elem_width_;
} BankConfig;

static inline const char *bank_name(const BankConfig *config) {
  return config->name_;
}
static inline uint32_t bank_base_addr(const BankConfig *config) {
  return config->base_addr_;
}
static inline uint32_t bank_row_num(const BankConfig *config) {
  return config->row_num_;
}
static inline uint32_t bank_elem_num(const BankConfig *config) {
  return config->elem_num_;
}
static inline uint32_t bank_elem_width(const BankConfig *config) {
  return config->elem_width_;
}

static inline uint32_t bank_row_width(const BankConfig *config) {
  return config->elem_num_ * config->elem_width_;
}

static inline uint32_t bank_row_bytes(const BankConfig *config) {
  return bank_row_width(config) / 8;
}

static inline uint32_t bank_total_size(const BankConfig *config) {
  return config->row_num_ * bank_row_bytes(config);
}

static inline uint32_t bank_addr(const BankConfig *config, uint32_t row) {
  return config->base_addr_ + row;
}

#define BANK_NUM 5
#define DIM 16

extern const BankConfig bank_configs[BANK_NUM];

#ifdef __cplusplus
}
#endif

#endif // _BANK_H_
