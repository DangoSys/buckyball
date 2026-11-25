# Linux programs installed into the poly rootfs, each with the hart it runs on.
set(IMAGE_PROGRAMS
  0 poly-hart0-ctest-intnorm_signed-linux
  0 poly-hart0-ctest-intnorm_zero-linux
  0 poly-hart0-ctest-intnorm_round-linux
  0 poly-hart0-ctest-intnorm_wide-linux
  0 poly-main-mlirtest-ball_intnorm_round-linux
  0 poly-main-mlirtest-bank_intnorm_round-linux
  0 poly-main-mlirtest-compiler_integer_norm-linux
  0 poly-main-mlirtest-compiler_reduce_sum-linux
  0 poly-main-ctest-program-linux
  0 poly-main-ctest-program_transfer-linux
  0 poly-main-ctest-design-controllers-linux
  0 poly-hart0-ctest-f32add_groups-linux
  0 poly-main-ctest-design-ant_shared-linux
  0 poly-main-ctest-design-ant_bank-linux
  0 poly-main-ctest-design-ant_parallel-linux
  0 poly-main-ctest-design-stream-linux
  PARENT_SCOPE)
