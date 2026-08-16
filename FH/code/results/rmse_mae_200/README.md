# RMSE and MAE evaluation on the fixed 200-slice test set

This directory contains the RMSE and MAE results added for the reviewer
revision. All predictions are clipped to `[0, 1]` before metric computation.
RMSE and MAE are computed independently for each slice and then averaged over
the same ordered 200-slice held-out test set used for the manuscript PSNR and
SSIM results.

- `quantitative_rmse_mae.csv` contains Sparse FBP, RED-CNN, Cascade, and DDF
  results for `S=2,4,8,12`.
- `dudotrans_S*/` contains the corresponding DuDoTrans per-slice and summary
  outputs. DuDoTrans uses native 512 x 512 preprocessing and exactly matched
  view counts. The `S=2,4,8` checkpoints are validation-selected reproductions;
  `S=12` uses the unchanged original 30-view checkpoint.
- `method_means.csv` is the combined manuscript-facing table.

The 256 x 256 methods are reproduced with:

```bash
cd FH/code
python experiments/run_quantitative_metrics.py \
  --sparse_factors 2,4,8,12 \
  --methods sparse_fbp,redcnn,cascade,ddf \
  --batch_size 2 \
  --output_csv results/rmse_mae_200/quantitative_rmse_mae.csv
```

DuDoTrans is reproduced with `dudotrans_autodl/test_dudotrans.py`, using the
checkpoint paths recorded in each `summary.csv` and seed 2026.
