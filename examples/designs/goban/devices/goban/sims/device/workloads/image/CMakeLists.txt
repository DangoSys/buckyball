# Linux programs installed into the goban rootfs, each with the hart it runs on.
# Goban's matmul threads are OS-invisible: the OS owns them and a program's
# kernel reaches them through the kernel driver.
set(IMAGE_PROGRAMS
  0 goban-matmul-ctest-design-matmul_threads-linux
  0 goban-matmul-ctest-design-stream-linux
  PARENT_SCOPE)
