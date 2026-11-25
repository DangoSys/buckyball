from _api_test_helper import run_bbdev_case

run_bbdev_case(
    "bbdev bebop-bemu --sim '--device toy --test" " toy_relu_test-singlecore-baremetal'"
)
run_bbdev_case("bbdev bebop-bemu --sim '--device toy --test toy-test-linux'")
