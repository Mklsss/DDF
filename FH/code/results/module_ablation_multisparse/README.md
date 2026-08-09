# Multi-sparsity module ablation results

These files report the completed module-level ablations at sparse factors
`S=2`, `S=4`, and `S=8`. Each validation-selected checkpoint was evaluated
once on the same ordered 200-slice held-out test set used by the main paper.

- `module_no_sine`: replaces the proposed Sine Fusion block with a
  parameter-free element-wise sum.
- `module_no_ct`: replaces the Cross Gating Block in CT Fusion with a
  parameter-free arithmetic average.

`summary.csv` contains validation and test-set means. `per_slice_metrics.csv`
contains the 1,200 slice-level PSNR/SSIM records used to reproduce the means.
The six JSON files under `tasks/` preserve the checkpoint epoch and evaluation
metadata for each run.

The results were produced by:

```bash
python experiments/evaluate_module_ablation_multisparse.py \
  --test_npz dataset/test_meiaonew.npz \
  --checkpoint_dir FH/code/weights/reviewer21_checkpoint_fixed_v3 \
  --output_dir FH/code/results/module_ablation_multisparse \
  --batch_size 1
```
