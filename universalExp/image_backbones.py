"""Image-domain backbones used by the universal DDF ablation experiments."""

from torch import nn


class RedCNN(nn.Module):
    """RED-CNN denoiser with the original five convolution/deconvolution pairs.

    Its interface is deliberately identical to DDF's NAFNet image backbone:
    ``(N, 1, 256, 256) -> (N, 1, 256, 256)``.
    """

    def __init__(self, channels=96):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be positive")
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=5)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5)
        self.tconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(channels, channels, kernel_size=5)
        self.tconv3 = nn.ConvTranspose2d(channels, channels, kernel_size=5)
        self.tconv4 = nn.ConvTranspose2d(channels, channels, kernel_size=5)
        self.tconv5 = nn.ConvTranspose2d(channels, 1, kernel_size=5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        residual_1 = image
        output = self.relu(self.conv1(image))
        output = self.relu(self.conv2(output))
        residual_2 = output
        output = self.relu(self.conv3(output))
        output = self.relu(self.conv4(output))
        residual_3 = output
        output = self.relu(self.conv5(output))
        output = self.tconv1(output) + residual_3
        output = self.tconv2(self.relu(output))
        output = self.tconv3(self.relu(output)) + residual_2
        output = self.tconv4(self.relu(output))
        output = self.tconv5(self.relu(output)) + residual_1
        return self.relu(output)
