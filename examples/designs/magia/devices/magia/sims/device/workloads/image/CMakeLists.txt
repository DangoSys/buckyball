# Linux programs installed into the magia rootfs, each with the hart it runs on.
set(IMAGE_PROGRAMS
  0 magia-hart0-ctest-gemm_service-linux
  PARENT_SCOPE)
set(IMAGE_DTS_OVERLAY "${CMAKE_CURRENT_SOURCE_DIR}/mesh-reserved.dts" PARENT_SCOPE)
