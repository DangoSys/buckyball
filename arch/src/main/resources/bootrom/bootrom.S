# Buckyball BootROM: validate mhartid and jump to the DRAM entry in M-mode.
# The SoC fills the two dwords at _config (ROM offsets 0x8 and 0x10) when it
# builds the ROM, so this image is the same for every chip. OpenSBI takes the
# DTB embedded in its fw_payload.

.section .text
.globl _entry
_entry:
  j       _start

  .balign 8
_config:
  .dword  0                 # 0x08: hart count
  .dword  0                 # 0x10: DRAM entry

_start:
  csrr    a0, mhartid
  la      t0, _config
  ld      t1, 0(t0)         # hart count
  bgeu    a0, t1, park      # an unknown hart never leaves the ROM
  li      a1, 0             # no boot DTB
  ld      t1, 8(t0)         # DRAM entry
  jr      t1

park:
  wfi
  j       park
