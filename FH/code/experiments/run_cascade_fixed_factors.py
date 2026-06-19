from ddf_experiment_lib import common_parser, run_fixed_factor_experiment


def main():
    parser = common_parser("Run cascade baseline experiments for fixed sparse-view factors.")
    args = parser.parse_args()
    run_fixed_factor_experiment(args, "cascade", "cascade", "Cascade", "cascade_fixed_factors.csv")


if __name__ == "__main__":
    main()
