from ddf_experiment_lib import common_parser, run_fixed_factor_experiment


def main():
    parser = common_parser("Run Sine Fusion ablation experiments.")
    parser.add_argument(
        "--variant",
        choices=("proposed", "simple"),
        default="proposed",
        help="proposed uses DDF gMLP Sine Fusion; simple uses element-wise addition.",
    )
    args = parser.parse_args()
    if args.variant == "proposed":
        run_fixed_factor_experiment(args, "ddf", "ddf", "DDF-SineFusion", "ablation_sine_fusion.csv")
    else:
        run_fixed_factor_experiment(
            args,
            "ddf_no_sine_fusion",
            "sine_no_fusion",
            "DDF-NoSineFusion",
            "ablation_sine_fusion.csv",
        )


if __name__ == "__main__":
    main()
