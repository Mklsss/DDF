# DDF Experiment Entrypoints

Run commands from `F:\junior2\paper2\FH\code`.

The default config is `experiments/config_default.json`. It points to the existing notebook data under
`FHinner/tnt_v1/data` and checks checkpoint files before evaluation. If a checkpoint for a sparse factor is missing,
the scripts print `checkpoint for S=... not found, please train first.` and skip that item.

## Evaluation

```powershell
python experiments/run_ddf_fixed_factors.py --sparse_factor 2,4,8,12 --batch_size 1 --device cuda:0
python experiments/run_cascade_fixed_factors.py --sparse_factor 2,4,8,12 --batch_size 1 --device cuda:0
python experiments/run_ablation_sine_fusion.py --variant proposed --sparse_factor 2,4,8,12
python experiments/run_ablation_sine_fusion.py --variant simple --sparse_factor 2,4,8,12
python experiments/run_ablation_ct_fusion.py --variant cgb --sparse_factor 2,4,8,12
python experiments/run_ablation_ct_fusion.py --variant conv --sparse_factor 2,4,8,12
```

## Visualization

```powershell
python experiments/run_visualization.py --sparse_factor 2,4,8,12 --test_sample_index 0 --roi 96,96,160,160
```

## Training Missing Checkpoints

These commands do not run automatically. Use them only after confirming the auxiliary FP files exist in `weights`.

```powershell
python experiments/run_ddf_fixed_factors.py --mode train --sparse_factor 2 --batch_size 3 --device cuda:0
python experiments/run_ddf_fixed_factors.py --mode train --sparse_factor 4 --batch_size 3 --device cuda:0
python experiments/run_ddf_fixed_factors.py --mode train --sparse_factor 8 --batch_size 3 --device cuda:0
python experiments/run_ddf_fixed_factors.py --mode train --sparse_factor 12 --batch_size 3 --device cuda:0

python experiments/run_cascade_fixed_factors.py --mode train --sparse_factor 2 --batch_size 3 --device cuda:0
python experiments/run_cascade_fixed_factors.py --mode train --sparse_factor 4 --batch_size 3 --device cuda:0
python experiments/run_cascade_fixed_factors.py --mode train --sparse_factor 8 --batch_size 3 --device cuda:0
python experiments/run_cascade_fixed_factors.py --mode train --sparse_factor 12 --batch_size 3 --device cuda:0

python experiments/run_ablation_sine_fusion.py --mode train --variant simple --sparse_factor 2,4,8,12 --batch_size 3
python experiments/run_ablation_ct_fusion.py --mode train --variant conv --sparse_factor 2,4,8,12 --batch_size 3
```

## Outputs

- `results/ddf_fixed_factors.csv`
- `results/cascade_fixed_factors.csv`
- `results/ablation_sine_fusion.csv`
- `results/ablation_ct_fusion.csv`
- `results/summary.csv`
- `fig/reconstruction_comparison_S2.pdf` and `.png`, similarly for S=4,8,12
