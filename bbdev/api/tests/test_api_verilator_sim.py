from _api_test_helper import run_bbdev_case

run_bbdev_case(
    "bbdev verilator --sim '--test toy-toy-ctest-mvin_mvout_test-baremetal --batch --config"
    " toy/devices/toy/configs/ToyDefaultConfig.toml --device toy'"
)
