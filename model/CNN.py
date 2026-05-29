import torch.nn as nn

class ConvBlock(nn.Module):
    '''
    Convolutinoal block for the CNN model.
    Consists of a convolutional layer, batch normalization, ReLU activation and max pooling

    :param nn: PyTorch neural network module
    :type nn: torch.nn
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class UNOplayer(nn.Module):
    '''
    CNN model for active player prediction

    :param nn: PyTorch neural network module
    :type nn: torch.nn
    '''
    def __init__(self, init_hidden_channels=16, kernel_size=3, padding=1, n_down=4):
        super().__init__()
        
        self.init_hidden_channels = init_hidden_channels
        self.kernel_size = kernel_size
        self.n_down = n_down
        
        if padding is None:
            padding = kernel_size // 2

        self.padding = padding

        hid_dims = [init_hidden_channels * (2**i) for i in range(n_down)]

        layers = []

        layers.append(
            ConvBlock(3, hid_dims[0], kernel_size=kernel_size, padding=padding)
        )
        
        for i in range(1, n_down):
            layers.append(
                ConvBlock(hid_dims[i-1], hid_dims[i], kernel_size=kernel_size, padding=padding)
                )
        
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hid_dims[-1], 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        return x