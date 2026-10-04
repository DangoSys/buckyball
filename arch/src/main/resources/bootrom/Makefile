# Buckyball BootROM. The image is position independent; the SoC places it at its ROM base.
CC      = riscv64-none-elf-gcc
OBJCOPY = riscv64-none-elf-objcopy
CFLAGS  = -march=rv64i_zicsr -mabi=lp64 -nostdlib -static -Wl,--build-id=none -Wl,-Ttext=0x10000

all: bootrom.rv64.img

bootrom.elf: bootrom.S
	$(CC) $(CFLAGS) $< -o $@

bootrom.rv64.img: bootrom.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f bootrom.elf bootrom.rv64.img
