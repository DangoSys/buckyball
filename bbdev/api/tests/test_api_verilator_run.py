from _api_test_helper import run_bbdev_case

run_bbdev_case(
    "bbdev verilator --run '--jobs 16 --test toy-toy-ctest-mvin_mvout_test-baremetal --config"
    " toy/devices/toy/configs/ToyDefaultConfig.toml --device toy --batch'"
)
