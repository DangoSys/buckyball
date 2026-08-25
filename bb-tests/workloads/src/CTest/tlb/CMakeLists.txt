add_custom_target(tlb-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET tlb-ctests-build)

foreach(BUCKYBALL_CTEST_TARGET IN LISTS BUCKYBALL_TARGETS)
  add_buckyball_ctests(
    tlb_bare_mode_test.c
    tlb_repeated_sfence_test.c
    tlb_sfence_flush_test.c
    tlb_sfence_remap_test.c
    tlb_sv39_4k_alias_test.c
    tlb_sv39_identity_test.c
    tlb_sv39_nonidentity_test.c
  )
endforeach()
