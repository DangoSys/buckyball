from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev verilator --run '--jobs 16 --binary ctest_vecunit_matmul_ones_singlecore-baremetal --config sims.verilator.BuckyballToyVerilatorConfig --batch'")
