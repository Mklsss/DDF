# Reviewer 1 scheduler sensitivity analysis

This directory contains the validation histories for a controlled 100-epoch
comparison of StepLR, cosine annealing, and ReduceLROnPlateau at S=12. All runs
use the same 1,600/200 training/validation split, seed 2026, shared initial
weights, Adam optimizer, and batch size 3. Values are validation metrics and
must not be compared directly with the held-out test-patient values in the main
quantitative table.

Run `experiments/plot_scheduler_comparison.py --input_dir
results/scheduler_r1_S12` to regenerate `summary.csv` and
`scheduler_curves.tex`. Compile the latter with a TeX Live installation that
contains PGFPlots to obtain the vector figure used in the manuscript.
