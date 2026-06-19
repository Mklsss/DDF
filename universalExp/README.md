# Universal DDF backbone experiment

本目录独立于原始实现，当前已实现完整 DDF 中以 P-CNN 或 P-Swin 替换投影域
`sin_angle` backbone，以及以 I-CNN（RED-CNN）或 I-Restor（Restormer）替换图像域
NAFNet 的变体；也包括 Mixed（P-Swin + RED-CNN）双域替换。完整实验矩阵、固定协议、当前验证状态、正式命令和结果登记规则统一维护在
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。
