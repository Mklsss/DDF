# Third-party code and assets

Repository-level licensing applies to the DDF authors' original contributions. It does not replace third-party copyright notices or license conditions, or grant rights to datasets, pretrained weights, or external assets. Preserve the copyright headers already present in source files.

## NAFNet and BasicSR

The following directories include NAFNet code with copyright headers identifying `megvii-model`, and `arch_util.py` identifies BasicSR-derived utilities:

- `FH/code/nafnet/`
- `FH/code/FHinner/tnt_v1/nafnet/`

Upstream: https://github.com/megvii-research/NAFNet

The upstream license file contains MIT terms for NAFNet and Apache-2.0 terms for BasicSR. A copy is retained in [third_party/NAFNet-LICENSE.txt](third_party/NAFNet-LICENSE.txt).

## Other references present in source files

These references record provenance visible in the current source; they are not a claim that every third-party file has been exhaustively audited or relicensed:

- `FH/code/model/maxim.py` and its nested copy reference [MPRNet](https://github.com/swz30/MPRNet).
- `dudotrans_autodl/modules/Unet_parts.py` references [Pytorch-UNet](https://github.com/xiaopeng-liao/Pytorch-UNet) and [U-Net-Pytorch-Unstructured-Buggy](https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy).
- `redcnn_autodl/measure.py` references [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).
- `redcnn_autodl/prep.py` references [an upstream augmentation snippet](https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d).

Consult the corresponding upstream sources and retained file headers for their terms before redistributing those components.

## Data and publications

The training/test archives and projection assets are not automatically covered by the software license. Follow the original data provider's access and reuse terms. The linked paper has its own publication license, which is separate from the source-code license.
