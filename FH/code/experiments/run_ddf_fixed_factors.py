from ddf_experiment_lib import common_parser, run_fixed_factor_experiment


def main():
    parser = common_parser("Run DDF main-result experiments for fixed sparse-view factors.")
    args = parser.parse_args()
    run_fixed_factor_experiment(args, "ddf", "ddf", "DDF", "ddf_fixed_factors.csv")


if __name__ == "__main__":
    main()
