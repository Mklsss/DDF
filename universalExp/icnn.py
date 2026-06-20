"""Image-domain RED-CNN used by the universal DDF backbone study."""

from torch import nn


class REDCNN(nn.Module):
    """Five-layer RED-CNN with long residual connections for 256x256 FBP images."""

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
        if image.ndim != 4 or image.shape[1:] != (1, 256, 256):
            raise ValueError(
                "REDCNN expects DDF FBP images with shape (N, 1, 256, 256), "
                f"received {tuple(image.shape)}"
            )
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
        # DDF's original NAFNet ``ct`` output is signed.  A final ReLU would
        # make RED-CNN incapable of representing that teacher and can leave it
        # permanently dead at zero during warm-start distillation.
        return self.tconv5(self.relu(output)) + residual_1
