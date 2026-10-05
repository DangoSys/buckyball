// PLIC priority, enable and threshold registers hold their values; with no
// source asserted nothing is pending, a claim returns zero and no external
// interrupt is raised.
#include "soc.h"

int main(void) {
  uint64_t context = 2 * read_csr(mhartid);
  PLIC_PRIORITY(1) = 5;
  if (PLIC_PRIORITY(1) != 5)
    return 1;
  PLIC_ENABLE(context) = 1u << 1;
  if (PLIC_ENABLE(context) != (1u << 1))
    return 2;
  PLIC_THRESHOLD(context) = 2;
  if (PLIC_THRESHOLD(context) != 2)
    return 3;
  if (PLIC_PENDING != 0)
    return 4;
  if (PLIC_CLAIM(context) != 0)
    return 5;
  PLIC_CLAIM(context) = 1;
  if (read_csr(mip) & MIP_MEIP)
    return 6;
  PLIC_ENABLE(context) = 0;
  PLIC_PRIORITY(1) = 0;
  return 0;
}
