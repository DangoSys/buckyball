#include <stdint.h>
#include <string.h>
#include <unistd.h>

extern const unsigned char _binary_mesh_bin_start[];
extern const unsigned char _binary_mesh_bin_end[];

int main(void) {
  for (unsigned round = 0; round < 2; ++round) {
    // Rewrite the entry through the CPU cache before the mesh fetches it.
    memcpy((void *)(uintptr_t)UINT32_C(0xcc000080), _binary_mesh_bin_start,
           _binary_mesh_bin_end - _binary_mesh_bin_start);
    *(volatile uint32_t *)(uintptr_t)UINT32_C(0xcc010000) = 0;
    asm volatile("fence rw, rw" : : : "memory");
    uintptr_t entry = UINT32_C(0xcc000080);
    asm volatile(".insn r 0x0b, 2, 0, x0, %0, x0" : : "r"(entry) : "memory");
    uint64_t status = 0;
    unsigned booted = 0;
    char buffer[64];
    unsigned count = 0;
    for (;;) {
      if (!booted && *(volatile uint32_t *)(uintptr_t)UINT32_C(0xcc010000)) {
        booted = 1;
        write(1, "MAGIA mesh booted\n", sizeof("MAGIA mesh booted\n") - 1);
      }
      uint64_t output;
      asm volatile(".insn r 0x0b, 4, 2, %0, x0, x0"
                   : "=r"(output)
                   :
                   : "memory");
      if (output & 0x100) {
        buffer[count++] = (char)output;
        if (count == sizeof(buffer)) {
          write(1, buffer, count);
          count = 0;
        }
      }
      if (status & 1) {
        if (!(output & 0x100))
          break;
      } else {
        asm volatile(".insn r 0x0b, 4, 1, %0, x0, x0"
                     : "=r"(status)
                     :
                     : "memory");
      }
    }
    if (count)
      write(1, buffer, count);
    if (!booted || status >> 32)
      return 1;
    char message[] = "MAGIA mesh round 0 passed\n";
    message[17] = (char)('0' + round);
    write(1, message, sizeof(message) - 1);
  }
  return 0;
}
