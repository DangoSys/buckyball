from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev verilator --sim '--binary toy_relu_test-singlecore-baremetal --batch --chip toy'")
