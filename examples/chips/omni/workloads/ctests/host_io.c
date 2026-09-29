#include <stdint.h>
#include <unistd.h>

int main(void) {
  for (;;) {
    uint64_t input;
    size_t offset = 0;
    while (offset < sizeof(input)) {
      ssize_t count = read(0, (char *)&input + offset, sizeof(input) - offset);
      if (count == 0)
        return offset == 0 ? 0 : 1;
      if (count < 0)
        return 2;
      offset += count;
    }
    if (input == UINT64_MAX)
      return 11;
    uint64_t output = input * input + 7;
    offset = 0;
    while (offset < sizeof(output)) {
      ssize_t count =
          write(1, (char *)&output + offset, sizeof(output) - offset);
      if (count <= 0)
        return 3;
      offset += count;
    }
  }
}
