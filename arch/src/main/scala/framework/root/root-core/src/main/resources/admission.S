.option norelax
.section .text.start,"ax",@progbits
.globl _start
_start:
  la t0,trap_handler
  csrw mtvec,t0
  li s0,0x10000000
  li t0,0x20400000
  csrw pmpaddr0,t0
  li t0,0x0f
  csrw pmpcfg0,t0
  li t0,0x8000000000080006
  csrw satp,t0
  li t0,3<<11
  csrc mstatus,t0
  li t0,(1<<11)|(1<<17)|(1<<18)|(1<<19)
  csrs mstatus,t0
  li s1,0x400040c0
  li t0,0x1111222233334444
  mv t1,s1
.globl admit_old_load
admit_old_load:
  ld t3,0(s1)
.globl admit_cmd0
admit_cmd0:
  .insn r 0x7b,3,0x11,zero,t0,t1
.globl admit_young_load
admit_young_load:
  ld t4,8(s1)
  li t5,0x0123456789abcdef
  bne t3,t5,fail
  li t5,0x8877665544332211
  bne t4,t5,fail
  li t0,(1<<17)|(1<<18)|(1<<19)
  csrc mstatus,t0
  li t0,0x20800000
  csrw pmpaddr0,t0
  li t0,0x0b
  csrw pmpcfg0,t0
  li t0,0x8000300000080009
  csrw satp,t0
  li t0,0x5555666677778888
  li t1,0x400040c8
.globl admit_cmd1
admit_cmd1:
  .insn r 0x7b,3,0x12,zero,t0,t1
.globl admit_fault
admit_fault:
  .insn r 0x7b,1,0x13,zero,t0,t1
.globl admit_after_fault
admit_after_fault:
  sd zero,0(s0)
1: j 1b
fail:
  li t0,1
  sd t0,0(s0)
2: j 2b
.section .text.trap,"ax",@progbits
.option push
.option norvc
trap_handler:
  csrr t0,mcause
  li t1,2
  bne t0,t1,fail
  csrr t0,mepc
  la t1,admit_fault
  bne t0,t1,fail
  addi t0,t0,4
  csrw mepc,t0
  mret
.option pop
.section .data,"aw",@progbits
.quad 0x0123456789abcdef,0x8877665544332211
.section .pt.root,"aw",@progbits
.zero 8
.quad 0x20001c01
.zero 4080
.section .pt.mid,"aw",@progbits
.quad 0x20002001
.zero 4088
.section .pt.leaf,"aw",@progbits
.zero 32
.quad 0x200010c7
.zero 4056
