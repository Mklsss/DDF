# DDF: A Dual-Domain Fusion Network for Sparse-View CT Reconstruction

Research code for **DDF**, a framework for sparse-view computed tomography (CT) reconstruction. DDF combines projection-domain and image-domain reconstruction with forward-projection feedback, Sine Fusion, and CT Fusion modules.

**Paper:** [DDF: A Dual-Domain Fusion Network for Sparse-View CT Reconstruction](https://doi.org/10.32604/cmes.2026.089588), *Computer Modeling in Engineering & Sciences*, 149(1), article 47, 2026.

This repository contains model implementations, training and evaluation scripts, baseline experiments, and ablation studies. It is intended for research and reproducibility.

## Repository guide

| Path | Contents |
| --- | --- |
| `FH/code/DDP_run_c{2,4,8,12}.py` | Original DDF implementations for the four sparsity factors; these legacy scripts also contain training code |
| `FH/code/experiments/train_ddf_clean.py` | Command-line training wrapper around the original DDF implementation |
| `FH/code/experiments/run_quantitative_metrics.py` | Quantitative evaluation including original DDF and baseline methods |
| `FH/code/experiments/` | Configurable experiments, module ablations, visualization, noise robustness, and computational-cost analysis |
| `universalExp/` | Backbone substitution experiments and their [experiment protocol](universalExp/EXPERIMENT_PLAN.md) |
| `dudotrans_autodl/` | [DuDoTrans baseline instructions](dudotrans_autodl/README.md) |
| `redcnn_autodl/` | [RED-CNN baseline instructions](redcnn_autodl/README.md) |
| `FH/code/results/` | Selected experiment summaries and statistical analyses |
| `dataset/` | Test archive tracked with Git LFS |

The original model and the generic experiment library have different default image backbones: the original DDF uses NAFNet width 32 and encoder blocks `[1, 1, 1, 28]`, while `experiments/config_default.json` uses width 16 and `[1, 1, 1, 1]`. Match the architecture, checkpoint, data split, and metric protocol when comparing results. The generic entry points are not interchangeable with original-model checkpoints.

## Installation

A CUDA-capable NVIDIA GPU is recommended for training. The existing server environment record uses Python 3.10, PyTorch 2.1.2, torchvision 0.16.2, and CUDA 11.8 wheels. The following is a starting environment for the original DDF workflow; a complete clean-install training run has not been validated as part of this documentation update.

Install [Git LFS](https://git-lfs.com/) before cloning. The data and projection matrices are large binary files; a checkout containing only LFS pointer files cannot run the experiments.

```bash
git lfs install
git clone https://github.com/Mklsss/DDF.git
cd DDF
git lfs pull

conda create -n ddf python=3.10 -y
conda activate ddf
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 "opencv-python<4.12" pillow matplotlib tqdm einops scipy scikit-image swanlab
```

For a different GPU/runtime, install a compatible PyTorch/torchvision pair using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). Baseline and backbone substitution experiments may require additional dependencies; consult their directory documentation. The `dudotrans_autodl/requirements_server.txt` file is an environment snapshot containing server-local paths, not a portable installation requirements file.

## Data and projection operators

The DDF loaders expect NumPy `.npz` archives with these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `sin357` | `(N, 360, 357)` | Full-view sinogram from which sparse views are selected |
| `ct_label` | `(N, 256, 256, 1)` | Corresponding reference CT images |

Use the preprocessing, normalization, and acquisition geometry corresponding to the supplied projection operators. The code assumes 360 full views, 357 detector bins, and 256 × 256 reconstructions. Sparsity factors `2`, `4`, `8`, and `12` correspond to 180, 90, 45, and 30 retained views.

The following assets are tracked with Git LFS:

```text
FH/code/FHinner/tnt_v1/data/train_meiaonew.npz
dataset/test_meiaonew.npz
FH/code/model/A_new.npz
FH/code/weights/index_fpnew.npy
FH/code/weights/data_fpnew.npy
```

The original scripts also load `./data/train_meiaonew.npz` and `./data/test_meiaonew.npz` relative to `FH/code`, including when the evaluation wrapper imports their model definitions. Prepare these paths from the repository root:

```bash
python -c "from pathlib import Path; import shutil; p=Path('FH/code/data'); p.mkdir(parents=True, exist_ok=True); shutil.copy2('FH/code/FHinner/tnt_v1/data/train_meiaonew.npz', p/'train_meiaonew.npz'); shutil.copy2('dataset/test_meiaonew.npz', p/'test_meiaonew.npz')"
```

This creates local copies and requires additional disk space. Source data originate from the 2016 Low-Dose CT Grand Challenge used in the paper; dataset access and reuse remain subject to the source provider's terms. A software license does not grant rights to third-party medical data.

## Original DDF training and evaluation

Run these commands from `FH/code`:

```bash
cd FH/code
python experiments/train_ddf_clean.py --help
python experiments/run_quantitative_metrics.py --help
```

Example training command for sparsity factor 12:

```bash
python experiments/train_ddf_clean.py --sparse_factor 12 --epochs 500 --batch_size 3 --val_batch_size 2 --device cuda:0 --train_count 1600 --val_count 200 --output_best weights/DDF_c12_best.pth --output_ckpt weights/DDF_c12_ckpt.pth
```

The explicit split uses the first 1600 samples of the training archive for training and the next 200 for validation, so that archive must contain at least 1800 samples. The separate test archive is used for final evaluation. The wrapper also supports initialization, resume, and scheduler options; consult `--help` before changing the protocol.

**Pretrained DDF checkpoints are not included in this Git repository.** Train the model or supply a compatible original-model checkpoint. The evaluation script searches for `weights/DDF_c{S}_best.pth` and `weights/DDF_c{S}_ckpt.pth`.

After preparing a checkpoint for S=12:

```bash
python experiments/run_quantitative_metrics.py --methods ddf --sparse_factors 12 --batch_size 2 --device cuda:0
```

To evaluate all four factors, prepare the corresponding checkpoints and use `--sparse_factors 2,4,8,12`. Results are written to `results/quantitative_results.csv`, including PSNR, SSIM, RMSE, MAE, sample counts, and checkpoint paths. Missing DDF checkpoints are skipped; a completed command alone does not mean all requested factors were evaluated.

## Additional experiments

See the [experiment entry-point guide](FH/code/experiments/README.md) for generic DDF/cascade training, module ablations, and visualization. These scripts accept a `--config` JSON override. Paths inside that configuration are resolved relative to `FH/code`; for the tracked test archive use:

```json
{
  "test_data": "../../dataset/test_meiaonew.npz"
}
```

Save this as a local JSON file and pass its path through `--config`. Configure the architecture and checkpoint names for the experiment you are actually running. The default generic configuration does not describe the full original DDF model.

For backbone substitutions, read the [universal experiment guide](universalExp/README.md) and [protocol](universalExp/EXPERIMENT_PLAN.md). Its example configuration files contain server-specific absolute paths that must be adapted to your machine.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{pan2026ddf,
  title   = {DDF: A Dual-Domain Fusion Network for Sparse-View CT Reconstruction},
  author  = {Pan, Ziqiao and Chen, Yuxi and Yang, Bo and Zhang, Legend and Zeng, Renee and Ma, Xiaoran and Liu, Zhixin and Wei, Wei and Xu, Guangyu},
  journal = {Computer Modeling in Engineering & Sciences},
  volume  = {149},
  number  = {1},
  pages   = {47},
  year    = {2026},
  doi     = {10.32604/cmes.2026.089588}
}
```

## Acknowledgments and third-party code

The image-domain backbone includes code from [NAFNet](https://github.com/megvii-research/NAFNet), with BasicSR-derived utilities. Other files contain upstream references to MPRNet, U-Net, and PyTorch SSIM implementations. Preserve existing copyright headers and consult [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for identified sources and the bundled NAFNet/BasicSR license text.

## License

The DDF authors' original code and documentation are provided under the custom [DDF Noncommercial Research License 1.0](LICENSE).

- You may use, copy, modify, and share the covered material for noncommercial scientific research, academic study, teaching, and reproducibility, subject to the license conditions.
- Commercial use, including commercial product development, internal business use, and paid services, requires separate written permission from the applicable copyright holders. Publishing a research paper does not by itself make a commercial activity noncommercial.
- Preserve the license and copyright notices and identify modified versions. Third-party components retain their original terms; datasets and other external assets are not relicensed by this grant.

**This is source-available research software with a noncommercial restriction, not an OSI-approved open-source license.** The full license text governs use.

中文说明：本项目原创代码允许非商业科研、论文复现、学习与教学，可在遵守许可的前提下修改和分享。未经相关权利人另行书面授权，不得用于商业产品、商业研发、企业内部业务或收费服务。第三方代码和数据仍遵守各自条款。

## Questions and contributions

Please use [GitHub Issues](https://github.com/Mklsss/DDF/issues) for reproducibility questions and bug reports. Include the script, sparsity factor, environment, relevant configuration, checkpoint identity, and error output. Contributions improving documentation, portability, and reproducibility are welcome.
