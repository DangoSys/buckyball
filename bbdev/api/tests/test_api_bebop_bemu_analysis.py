from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev bebop-bemu --analysis '--chip pebble --log-dir fixtures/bemu-analysis --itrace --mtrace'")  # fmt: skip
