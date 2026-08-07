# Original-checkpoint DuDoTrans statistics

This directory contains the slice-level data used for the DuDoTrans entries
and paired tests in the revised manuscript. No DuDoTrans checkpoint was
fine-tuned, converted, or otherwise modified.

- `dudotrans_S4_view090.csv`: native 90-view checkpoint output on 200 slices.
- `dudotrans_S12_view030.csv`: native 30-view checkpoint output on 200 slices.
- `per_slice_metrics.csv`: paper-method metrics aligned by sparse factor and
  test-slice index.
- `method_means.csv`: aggregate PSNR and SSIM values reported in the paper.
- `paired_significance.csv`: two-sided paired Wilcoxon tests, Holm-adjusted
  p-values, rank-biserial effects, and 10,000-resample bootstrap intervals.
- `metadata.json`: source paths, SHA-256 hashes, protocol, and checkpoint
  availability notes.

The archived 120-view and 60-view files are truncated and cannot be loaded;
they are not replaced by a different checkpoint. They also do not match the
paper's 180-view and 45-view acquisition settings. Consequently, DuDoTrans is
reported only for the exactly matched, loadable S=4 and S=12 settings.

The merged statistics can be regenerated from this directory:

```bash
python FH/code/experiments/analyse_original_dudotrans.py \
  --base_metrics FH/code/results/statistical_significance_original_dudotrans/per_slice_metrics.csv \
  --dudotrans 4:90:FH/code/results/statistical_significance_original_dudotrans/dudotrans_S4_view090.csv:/path/to/view_090/epoch_019_iter_001799.pth.tar \
  --dudotrans 12:30:FH/code/results/statistical_significance_original_dudotrans/dudotrans_S12_view030.csv:/path/to/view_030/epoch_019_iter_001799.pth.tar \
  --output_dir FH/code/results/statistical_significance_original_dudotrans/recomputed
```
