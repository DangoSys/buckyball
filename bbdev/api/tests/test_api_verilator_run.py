from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev verilator --run '--jobs 16 --binary toy-toy-ctest-mvin_mvout_test-baremetal --chip toy --batch'")
