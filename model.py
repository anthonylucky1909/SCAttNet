import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = self.ca(out) * out  # Channel attention
        out = self.sa(out) * out  # Spatial attention
        return residual + out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

class SuperResolutionNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=16, base_channels=64):
        super().__init__()
        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 9, padding=4),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels, 5, padding=2),
            nn.PReLU()
        )
        # Residual blocks
        self.body = nn.Sequential(
            *[ResidualAttentionBlock(base_channels) for _ in range(n_blocks)]
        )
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.PReLU(),
            nn.Conv2d(base_channels*2, base_channels, 1)
        )
        # Progressive upsampling
        self.upsample = nn.Sequential(
            UpsampleBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.PReLU(),
            UpsampleBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.PReLU()
        )
        # Reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, out_channels, 9, padding=4)
        )
        # Learned skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x_skip = self.skip(x)
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.fusion(x)
        x += residual
        x = self.upsample(x)
        x = self.tail(x)
        return x + x_skip

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_maps=(64, 64, 128, 128, 256, 256, 512, 512)):
        super().__init__()
        # Feature extraction backbone
        conv_blocks = []
        current_channels = in_channels
        for idx, num_filters in enumerate(feature_maps):
            conv_blocks.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    stride=1 + idx % 2,  # Alternate stride 1 and 2
                    padding=1,
                    use_act=True
                )
            )
            current_channels = num_filters
        self.feature_extractor = nn.Sequential(*conv_blocks)
        # Classifier head (adapted for 128x128 input)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:25].eval()
        self.loss = nn.MSELoss()
        
    def forward(self, first, second):
        vgg_first = self.vgg(first)
        vgg_second = self.vgg(second)
        perceptual_loss = self.loss(vgg_first, vgg_second)
        return perceptual_loss
    

# if __name__ == "__main__":
#     model = SuperResolutionNet()
#     print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


#     input_tensor = torch.randn(1, 3, 64, 64)
#     output_tensor = model(input_tensor)

#     print(f"\nInput shape: {input_tensor.shape}")
#     print(f"Output shape: {output_tensor.shape}")

#     assert output_tensor.shape == (1, 3, 256, 256), "Upscaling failed!"
#     print("\nTest passed - model correctly scales 64×64 → 256×256")