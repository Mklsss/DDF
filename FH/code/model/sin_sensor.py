import torch
import torch.nn as nn
import torch.nn.functional as F


class sin_sensor(nn.Module):
    def __init__(self, num_team, sample_sensor, target_sensor, angle, scale=1000):
        super().__init__()
        self.target_sensor = int(target_sensor)
        self.scale = float(scale)
        hidden = max(16, int(num_team))
        self.fuse = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Expected x shape: [B, sensor, angle]
        b, sensor, angle = x.shape
        x = x.transpose(1, 2).reshape(b * angle, 1, sensor)
        x = F.interpolate(x, size=self.target_sensor, mode="linear", align_corners=False)
        x = self.fuse(x)
        x = x.reshape(b, angle, self.target_sensor).transpose(1, 2)
        return x
