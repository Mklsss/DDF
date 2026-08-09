# Updated DuDoTrans evaluation and paired statistics

This directory contains the fixed 200-slice test results used in the revised manuscript. DuDoTrans uses exactly matched view counts and its native 512 x 512 preprocessing. The S=2 (180-view), S=4 (90-view), and S=8 (45-view) models are validation-selected reproduction checkpoints; S=12 uses the unchanged original 30-view checkpoint.

Files:

- `dudotrans_S2_view180.csv`, `dudotrans_S4_view090.csv`, `dudotrans_S8_view045.csv`, and `dudotrans_S12_view030.csv`: ordered per-slice DuDoTrans metrics.
- `per_slice_metrics.csv`: aligned metrics for all reported methods.
- `method_means.csv`: manuscript table means.
- `paired_significance.csv`: two-sided paired Wilcoxon tests, Holm adjustment within each sparse-factor/metric family, rank-biserial effects, and 10,000-resample paired bootstrap confidence intervals.
- `metadata.json`: dataset, checkpoint, hash, and analysis provenance.

The analysis is reproduced with `FH/code/experiments/analyse_original_dudotrans.py`, passing the four per-slice CSV/checkpoint specifications and bootstrap seed 2026. The S=2 checkpoint was selected at epoch 42 with a validation PSNR of 37.5322 dB; its SHA-256 and full provenance are recorded in `metadata.json`.
