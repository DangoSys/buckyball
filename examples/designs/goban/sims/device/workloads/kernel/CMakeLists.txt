# Linux programs installed into the goban rootfs, each with the hart it runs on.
# Goban's matmul threads are OS-invisible: their tests are engine tests, not
# Linux programs, so Linux here only boots.
set(BUCKYBALL_KERNEL_PROGRAMS
  PARENT_SCOPE)
