#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <unistd.h>

int main(void) {
  if (write(1, (void *)0x80000000ULL, 1) != -1 || errno != EFAULT)
    return 7;
  uint32_t seed;
  if (getrandom(&seed, sizeof(seed), 0) != sizeof(seed))
    return 1;
  for (unsigned iteration = 0; iteration < 32; ++iteration) {
    size_t padding = (1 + ((seed >> (iteration % 24)) & 255)) * 4096;
    unsigned char *allocation = malloc(padding + 8192);
    if (!allocation)
      return 2;
    unsigned char *data = allocation + padding;
    for (unsigned i = 0; i < 8192; ++i)
      data[i] = (seed + iteration + i) & 255;
    int fd = open("process.bin", O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0 || write(fd, data, 8192) != 8192)
      return 3;
    memset(data, 0, 8192);
    if (lseek(fd, 0, SEEK_SET) != 0 || read(fd, data, 8192) != 8192)
      return 4;
    for (unsigned i = 0; i < 8192; ++i)
      if (data[i] != ((seed + iteration + i) & 255))
        return 5;
    if (close(fd))
      return 6;
    free(allocation);
  }
  puts("OMNI PROCESS MEMORY AND FILE ISOLATION PASSED");
  return 0;
}
