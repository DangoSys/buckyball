from _api_test_helper import run_bbdev_case

run_bbdev_case(
    "bbdev bebop-verilator --sim '--test toy_relu_test-singlecore-baremetal"
    " --device toy"
    " --itrace --mtrace --pmctrace --ctrace --banktrace'"
)
