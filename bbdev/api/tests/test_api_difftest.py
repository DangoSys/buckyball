from _api_test_helper import run_bbdev_case


run_bbdev_case(
    "bbdev difftest --run "
    "'--chip toy --config sims.verilator.BuckyballToyVerilatorConfig "
    "--binary toy_im2col_k3_test-singlecore-baremetal --no-wave'"
)
