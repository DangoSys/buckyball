from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-bemu --sim '--chip toy --binary toy_relu_test-singlecore-baremetal'")
run_bbdev_case("bbdev bebop-bemu --sim '--chip toy --binary toy_relu_test-linux --pk'")
