from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-verilator --run '--diff --no-wave --jobs 16 --binary toy_im2col_k3_test-singlecore-baremetal --config sims.verilator.BuckyballToyVerilatorConfig'")
