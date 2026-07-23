from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-verilator --run '--jobs 16 --binary vecunit_matmul_ones-singlecore-baremetal --config sims.verilator.BuckyballToyVerilatorConfig --batch'")
