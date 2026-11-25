# The start of a thread program alone: entry(argument) runs the task, and its
# return value is the completion value the environment reports: the status in
# bits 31:0 (0 passes).
.section .text.start,"ax",@progbits
.globl _start
.type _start,@function
_start:
  call entry
  li a7, 0
  ecall
.size _start, .-_start
