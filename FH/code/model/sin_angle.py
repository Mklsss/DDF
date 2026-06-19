import torch
import torch.nn as nn


class AttentionPos(nn.Module):
    def __init__(self, dim, angle, num_heads=7, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.angle = angle
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk = nn.Linear(dim * 4, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj1 = nn.Linear(dim, dim * 3)
        self.proj2 = nn.Linear(dim * 3, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def _build_positional(self, batch_size, n, c, device):
        sensor_center = (c - 1) / 2.0
        beta = (torch.arange(c, device=device, dtype=torch.float32) - sensor_center) / max(c, 1)
        beta = torch.atan(beta).view(1, 1, c).expand(batch_size, n, c)

        angle_val = torch.sin(torch.arange(n, device=device, dtype=torch.float32) / max(n, 1) * torch.pi)
        angle_val = angle_val.view(1, n, 1).expand(batch_size, n, c)

        mask = torch.zeros((batch_size, n, c), device=device, dtype=torch.float32)
        step = max(1, int(360 / max(self.angle, 1)))
        mask[:, ::step, :] = 1.0
        return angle_val, beta, mask

    def forward(self, x):
        b, n, c = x.shape
        angle_val, beta, mask = self._build_positional(b, n, c, x.device)
        x2 = self.norm1(x)
        qk = torch.cat((x2, angle_val, beta, mask), dim=2)
        qk = self.qk(qk).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(x2).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x2 = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x2 = self.proj(x2)
        x2 = x2 + x

        x3 = self.norm2(x2)
        x3 = self.proj1(x3)
        x3 = self.act(x3)
        x3 = self.proj2(x3)
        return x3 + x2


class Attention(nn.Module):
    def __init__(self, dim, num_heads=7, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj1 = nn.Linear(dim, dim * 3)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(dim * 3, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        b, n, c = x.shape
        x2 = self.norm1(x)
        qkv = self.qkv(x2).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x2 = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x2 = self.proj(x2)
        x2 = x2 + x

        x3 = self.norm2(x2)
        x3 = self.proj1(x3)
        x3 = self.act(x3)
        x3 = self.proj2(x3)
        return x3 + x2


class sin_angle(nn.Module):
    def __init__(self, num_sensor, angle, num_heads=7):
        super().__init__()
        self.attn_pos = AttentionPos(num_sensor, angle=angle, num_heads=num_heads)
        self.attn1 = Attention(num_sensor, num_heads=num_heads)
        self.attn2 = Attention(num_sensor, num_heads=num_heads)
        self.attn3 = Attention(num_sensor, num_heads=num_heads)
        self.attn4 = Attention(num_sensor, num_heads=num_heads)
        self.act = nn.ReLU()

    def forward(self, x):
        x_i = self.attn_pos(x)
        x_i = self.attn1(x_i)
        x_i = self.attn2(x_i)
        x_i = self.attn3(x_i)
        x_i = self.attn4(x_i) + x
        return self.act(x_i)
