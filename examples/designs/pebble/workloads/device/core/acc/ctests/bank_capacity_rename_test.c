#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>

_Static_assert(CORE_VIRTUAL_BANK_COUNT >= THREAD_MEM_BANK_NUM,
               "Test requires one virtual ID per physical bank");
static uint8_t input[THREAD_MEM_BANK_NUM][THREAD_MEM_BANK_WIDTH / 8]
    __attribute__((aligned(64)));
static uint8_t output[THREAD_MEM_BANK_NUM][THREAD_MEM_BANK_WIDTH / 8]
    __attribute__((aligned(64)));

int main(void) {
  for (unsigned bank = 0; bank < THREAD_MEM_BANK_NUM; ++bank)
    bb_mem_alloc(bank, 1, 1);
  for (unsigned round = 0; round < 3; ++round) {
    for (unsigned bank = 0; bank < THREAD_MEM_BANK_NUM; ++bank)
      for (unsigned byte = 0; byte < THREAD_MEM_BANK_WIDTH / 8; ++byte) {
        input[bank][byte] = bank * 7 + byte + round * 31;
        output[bank][byte] = 0;
      }
    for (unsigned bank = 0; bank < THREAD_MEM_BANK_NUM; ++bank)
      bb_mvin((uintptr_t)input[bank], bank, 1, 1);
    for (unsigned bank = 0; bank < THREAD_MEM_BANK_NUM; ++bank)
      bb_mvout((uintptr_t)output[bank], bank, 1, 1);
    for (unsigned bank = 0; bank < THREAD_MEM_BANK_NUM; ++bank)
      for (unsigned byte = 0; byte < THREAD_MEM_BANK_WIDTH / 8; ++byte)
        if (output[bank][byte] != input[bank][byte]) {
          printf("capacity rename mismatch round=%u bank=%u byte=%u\n", round,
                 bank, byte);
          return 1;
        }
  }
  for (unsigned bank = 0; bank < THREAD_MEM_BANK_NUM; ++bank)
    bb_mem_release(bank);
  puts("Bank capacity rename PASS");
  return 0;
}
