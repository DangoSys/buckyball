add_buckyball_ctests(
  smatmul_bias_accumulate_test.c
  smatmul_1x16x64_test.c
  smatmul_8x16_test.c
  smatmul_16x16_test.c
  smatmul_16x16x32_test.c
  smatmul_nondefault_banks_test.c
)
file(STRINGS "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h" F32_ISA
     REGEX "^#define BB_FUNC7_SMATMUL_F32 ")
if(F32_ISA)
  add_buckyball_ctest(smatmul_f32_test.c)
endif()
