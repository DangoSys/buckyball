from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-verilator --build '--jobs 16 --config sims.verilator.BuckyballToyVerilatorConfig'")
