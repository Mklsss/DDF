from ddf_experiment_lib import common_parser, run_fixed_factor_experiment


def main():
    parser = common_parser("Run CT Fusion ablation experiments.")
    parser.add_argument(
        "--variant",
        choices=("cgb", "conv"),
        default="cgb",
        help="cgb uses CrossGatingBlock; conv uses 0.5 * (CT_nd + CT_mid).",
    )
    args = parser.parse_args()
    if args.variant == "cgb":
        run_fixed_factor_experiment(args, "ddf", "ddf", "DDF-CGBFusion", "ablation_ct_fusion.csv")
    else:
        run_fixed_factor_experiment(
            args,
            "ddf_ct_conv_fusion",
            "ct_conv_fusion",
            "DDF-ConvCTFusion",
            "ablation_ct_fusion.csv",
        )


if __name__ == "__main__":
    main()
