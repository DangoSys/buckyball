from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev verilator --sim '--binary toy_vecunit_matmul_ones-singlecore-baremetal --batch --config sims.verilator.BuckyballToyVerilatorConfig'")
