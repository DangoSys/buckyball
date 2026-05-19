from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-verilator --batch '--config sims.verilator.BuckyballToyVerilatorConfig --test elf-tests'")
