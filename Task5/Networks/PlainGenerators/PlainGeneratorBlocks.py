import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import torch.nn as nn

cnnBias = True  # Global flag for convolution bias

# Convolutional block with optional normalization
class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, normalization=None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride size for the convolution.
            padding (int): Amount of padding.
            normalization (str): Type of normalization ('batchnorm', 'groupnorm', 'instancenorm', or None).
        """
        super(Convblock, self).__init__()
        ops = []

        # Apply LeakyReLU activation with negative slope
        ops.append(nn.LeakyReLU(negative_slope=0.2))
        
        # Define 2D convolutional layer
        ops.append(nn.Conv2d(in_channels=in_channels, 
                             out_channels=out_channels,
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding))
        
        # Add normalization layers based on the type provided
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=out_channels))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(out_channels))
        elif normalization != 'none':
            assert False  # Raise an error for unsupported normalization types

        # Create a sequential layer using the defined ops
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        # Pass input through the convolutional block
        x = self.conv(x)
        return x

# Deconvolutional block with optional normalization and dropout
class Deconvblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, normalization=None, dropout=True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the kernel for transposed convolution.
            stride (int): Stride size for the transposed convolution.
            padding (int): Padding for the transposed convolution.
            normalization (str): Type of normalization ('batchnorm', 'instancenorm', or None).
            dropout (bool): Whether to apply dropout.
        """
        super(Deconvblock, self).__init__()

        ops = []
        # Apply ReLU activation
        ops.append(nn.ReLU(inplace=True))

        # Perform transposed convolution (deconvolution)
        ops.append(nn.ConvTranspose2d(in_channels, out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1))

        # Add normalization layers based on the type provided
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(out_channels))
        elif normalization != 'none':
            assert False  # Raise an error for unsupported normalization types

        # Optionally add dropout to the layer
        if dropout:
            ops.append(nn.Dropout2d(p=0.5, inplace=False))

        # Create a sequential layer using the defined ops
        self.deconv = nn.Sequential(*ops)

    def forward(self, x):
        # Pass input through the deconvolutional block
        x = self.deconv(x)
        return x

# Residual convolutional block with optional normalization
class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mixer='ResidualMixer', normalization=None):
        """
        Args:
            n_stages (int): Number of stages in the residual block.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride size for the convolution.
            padding (str or int): Padding for the convolution ('same' or custom value).
            mixer (str): Type of mixer (default is 'ResidualMixer').
            normalization (str): Type of normalization ('batchnorm', 'instancenorm', or None).
        """
        super(ResidualConvBlock, self).__init__()

        # Handle padding logic: 'same' means padding to preserve the input dimensions
        if padding.lower() == 'same':
            padding = kernel_size // 2
        else:
            padding = 0

        ops = []
        # Define multiple stages of the residual block
        for i in range(n_stages):
            # First stage uses the given input channels
            if i == 0:
                input_channel = in_channels
            else:
                input_channel = out_channels

            # Add convolutional layer
            ops.append(nn.Conv2d(input_channel, out_channels, kernel_size, stride, padding))
            
            # Add normalization if specified
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(out_channels))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(out_channels))
            else:
                assert False  # Raise error for unsupported normalization types

            # Apply ReLU activation between the layers, except at the last stage
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        # Create a sequential layer using the defined ops
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        # Add the input (skip connection) to the output of the convolutional block
        x = self.conv(x) + x
        return x