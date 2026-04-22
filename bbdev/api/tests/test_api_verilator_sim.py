from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev verilator --sim '--binary ctest_vecunit_matmul_ones_singlecore-baremetal --batch'")
