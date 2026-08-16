# Multi-sparsity backbone ablation

This directory records the backbone-substitution ablation results reported in the revised manuscript.

- `summary.csv` contains the PSNR/SSIM values on the fixed 200-slice held-out test set. The S=2, S=4, and S=8 entries were recomputed from the validation-selected `reviewer21_corrected_v2` checkpoints. The S=12 entries are the previously reported matched backbone-ablation results.
- `checkpoint_metadata.csv` records the best validation epoch, validation PSNR, checkpoint size, and SHA-256 for every newly evaluated S=2/S=4/S=8 checkpoint.
- Sparse factors S=2, S=4, S=8, and S=12 correspond to 180, 90, 45, and 30 projection views, respectively.
- All newly evaluated checkpoints use `dataset/test_meiaonew.npz` with 200 ordered test slices and seed 2026.

The checkpoints are not committed because they range from approximately 66 MB to 494 MB. Their hashes provide provenance for the reported results.
