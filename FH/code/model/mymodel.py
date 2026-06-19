from functools import partial
import torch
import torch.nn as nn
from .sin_sensor import sin_sensor
from .sin_angle import sin_angle
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# from .FBP import FbpLayer

class mymodel(nn.Module):
    def __init__(self, num_team, num_sensor, target_sensor, angle,
                 num_heads=7,
                 sample_sensor=261):  # num_team大trans一组20个，num_sensor裁剪的260个，targetsensor，复原为420个，angle为180°
        super().__init__()
        self.sin_sensor_lay1 = sin_sensor(num_team, sample_sensor, target_sensor, angle, scale=1000)
        self.sin_angle_lay1 = sin_angle(num_sensor, angle, num_heads)
        self.sample = int(360/angle)
        self.sin_sensor_lay2 = sin_sensor(num_team, sample_sensor, target_sensor, 360, scale=1000)
        self.sin_angle_lay2 = sin_angle(605, angle, num_heads=5)
        self.conlay31 = nn.Conv2d(1, 1, self.sample+1, 1, padding=int(self.sample/2))
        self.conlay32 = nn.Conv2d(1, 1, 3, 1, padding=1)
        self.act = nn.Sigmoid()
        # self.fbp=FbpLayer()

    def forward(self, x):
        B, angle, _ = x.shape  # 3*180*357
        x = x.transpose(2, 1)
        x_cat = torch.zeros((B, 42, angle), device=x.device)  # 3*42*180

        xlay1_1 = self.sin_sensor_lay1(x)  # 3*520*180
        x_up = torch.cat((x_cat, xlay1_1, x_cat), dim=1)  # 3*604*180
        center_idx = x.shape[1] // 2
        xlay1_1 = torch.cat((x_up[:, 0:302, :], x[:, center_idx, :].unsqueeze(1), x_up[:, 302:, :]), dim=1)  # 3*605*180

        xlay1_2 = self.sin_angle_lay1(x)  # 3*360*357

        xlay2_1 = self.sin_angle_lay2(xlay1_1)  # 3*360*605

        xlay1_2 = xlay1_2.transpose(2, 1)  # 3*357*360
        xlay2_2 = self.sin_sensor_lay2(xlay1_2)  # 3*520*360
        x_cat = torch.zeros((B, 42, 360), device=x.device)
        x_up = torch.cat((x_cat, xlay2_2, x_cat), dim=1)
        center_idx_2 = xlay1_2.shape[1] // 2
        xlay2_2 = torch.cat((x_up[:, 0:302, :], xlay1_2[:, center_idx_2, :].unsqueeze(1), x_up[:, 302:, :]), dim=1)

        x = (xlay2_1.transpose(2, 1) + xlay2_2) / 2

        return xlay1_1, xlay1_2, x  # 3*605*360

