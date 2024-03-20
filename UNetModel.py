import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.conv(x)


class UNet(nn.Module):
    """
    UNet neural network class.
    in_channels - number of channels in input images.
    out_classes - number of channels for output mask.
    features - each number adds a layer with corresponding number of channels in down and up part.
    """

    def __init__(
            self, in_channels=3, out_channels=1, features=(32, 64, 128, 256),
    ):
        super(UNet, self).__init__()
        # Creating module lists for layers so we can call them
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Performing the downsampling
        for down in self.downs:
            x = down(x)

            # Saving the skip connections
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reversing the list,
        # because we will feed skip connections starting from the last one
        skip_connections = skip_connections[::-1]

        # Performing the upsampling
        for idx in range(0, len(self.ups), 2):
            # Feed data to Transpose Conv layer
            x = self.ups[idx](x)

            # Add the corresponding skip connection
            skip_connection = skip_connections[idx//2]

            # Concatenating the skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Feeding the result to the Double Conv layer
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def load_weights(self, model_weights_path: str):
        model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
        self.load_state_dict(model_weights)
