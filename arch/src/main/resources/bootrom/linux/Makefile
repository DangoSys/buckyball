bootrom_img = bootrom.rv64.img

GCC = riscv64-unknown-elf-gcc
CFLAGS = -mabi=lp64 -march=rv64ima_zicsr
OBJCOPY = riscv64-unknown-elf-objcopy

all: $(bootrom_img)

%.img: %.elf
	$(OBJCOPY) -O binary --change-addresses=-0x10000 $< $@

%.rv64.elf: %.S linker.ld
	$(GCC) $(CFLAGS) -Tlinker.ld $< -nostdlib -static -o $@

clean:
	rm -f *.elf *.img
