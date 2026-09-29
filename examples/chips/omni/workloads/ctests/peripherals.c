#include <interconnect.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
  if (link_read(LINK_CHIP_ID) != 4)
    return 0;
  volatile uint8_t *uart = (volatile uint8_t *)0x60020000ULL;
  puts("OMNI VM READY");
  while (!(uart[5] & 1)) {
  }
  if (uart[4] != 'g')
    return 1;
  volatile uint32_t *vga = (volatile uint32_t *)0x11000000ULL;
  volatile uint32_t *pixels = (volatile uint32_t *)0x11001000ULL;
  unsigned width = vga[0], height = vga[1];
  for (unsigned y = 0; y < height; ++y)
    for (unsigned x = 0; x < width; ++x)
      pixels[y * width + x] =
          ((x * 255 / width) << 16) | ((y * 255 / height) << 8) | 0x40;
  vga[3] = 1;
  volatile uint32_t *speaker = (volatile uint32_t *)0x10002000ULL;
  volatile int16_t *sample = (volatile int16_t *)0x10002004ULL;
  for (unsigned i = 0; i < 256; ++i)
    *sample = ((int)(i % 64) - 32) * 256;
  speaker[3] = 1;
  while (speaker[0]) {
  }
  speaker[3] = 0;
  puts("OMNI VM TEXT FRAME AUDIO PASSED");
  return 0;
}
