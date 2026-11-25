from _api_test_helper import run_bbdev_case

run_bbdev_case(
    "bbdev bebop-bemu --analysis '--device toy"
    " --run 2026-11-21-00-00-toy-bemu-device-ToyDefaultConfig-toy-toy_relu_test-singlecore-baremetal"
    " --itrace --mtrace'"
)
