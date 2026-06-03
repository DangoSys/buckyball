from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev kernel --build")
run_bbdev_case("bbdev kernel --build '--visible-hart-count 64 --total-hart-count 256'")
