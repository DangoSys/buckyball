from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-verilator --sim '--binary ctest_vecunit_matmul_ones_singlecore-baremetal --batch --config sims.verilator.BuckyballToyVerilatorConfig --itrace --mtrace --pmctrace --ctrace --banktrace''")
