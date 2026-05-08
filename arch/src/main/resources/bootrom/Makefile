# Buckyball P2E Bootrom Makefile
# Input: p2e.S
# Output: p2e.bootrom.img

# Toolchain (will be provided by nix develop)
CC = riscv64-none-elf-gcc
OBJCOPY = riscv64-none-elf-objcopy

# Compiler flags
CFLAGS = -march=rv64ima_zicsr -mabi=lp64 -nostdlib -static
LDFLAGS = -Wl,--build-id=none -Wl,-Ttext=0x10000

# Files
SRC = p2e.S
ELF = p2e.bootrom.elf
IMG = p2e.bootrom.img

.PHONY: all clean

all: $(IMG)

$(ELF): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@

$(IMG): $(ELF)
	$(OBJCOPY) -O binary $< $@
	@echo "Generated $@"
	@ls -lh $@

clean:
	rm -f $(ELF) $(IMG)
