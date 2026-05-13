# Buckyball Bootrom Makefile
# Input: bootrom.S
# Output: bootrom.img

# Toolchain (will be provided by nix develop)
CC = riscv64-none-elf-gcc
OBJCOPY = riscv64-none-elf-objcopy

# Compiler flags
CFLAGS = -march=rv64ima_zicsr -mabi=lp64 -nostdlib -static
LDFLAGS = -Wl,--build-id=none -Wl,-Ttext=0x10000

# Files
SRC = bootrom.S
ELF = bootrom.elf
IMG = bootrom.img

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
