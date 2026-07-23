from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-bemu --sim '--chip toy --binary vecunit_matmul_ones-singlecore-baremetal'")
run_bbdev_case("bbdev bebop-bemu --sim '--chip toy --binary ctest_vecunit_matmul_ones-linux --pk'")
