from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-verilator --sim '--binary vecunit_matmul_ones-singlecore-baremetal --batch --config sims.verilator.BuckyballToyVerilatorConfig --itrace --mtrace --pmctrace --ctrace --banktrace'")
