import sys
import torch.nn as nn

# Add the path for additional modules (ensure these paths are correct in your project)
sys.path.append('./')
from Networks.PlainGenerators.PlainGeneratorBlocks import Convblock

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, normalization, loadedCategoryLength=80, input_channels=1, generator_dim=64):
        """
        Args:
            normalization: The type of normalization to apply (batchnorm, instancenorm, etc.).
            loadedCategoryLength: Number of output categories for classification.
            input_channels: Number of input channels (default: 1 for grayscale images).
            generator_dim: The base number of feature maps for the encoder (default: 64).
        """
        super(Encoder, self).__init__()
        
        # Define convolutional and residual blocks for encoding the input
        self.normalization = normalization
        
        # Input layer: Convolution (input_channels -> generator_dim) [e.g., 1x64x64 -> 64x32x32]
        self.encodingBlock0 = nn.Conv2d(input_channels, generator_dim, kernel_size=5, stride=2, padding=2) 
        
        # Additional encoding blocks that progressively downsample and increase the number of feature maps
        self.encodingBlock1 = Convblock(generator_dim, generator_dim * 2, normalization=normalization)  # 128x16x16
        self.encodingBlock2 = Convblock(generator_dim * 2, generator_dim * 4, normalization=normalization)  # 256x8x8
        self.encodingBlock3 = Convblock(generator_dim * 4, generator_dim * 8, normalization=normalization)  # 512x4x4
        self.encodingBlock4 = Convblock(generator_dim * 8, generator_dim * 8, normalization=normalization)  # 512x2x2
        self.encodingBlock5 = Convblock(generator_dim * 8, generator_dim * 8, normalization=normalization)  # 512x1x1
        
        # A fully connected layer for classification (flattening the encoded features)
        self.category = nn.Linear(generator_dim * 8, loadedCategoryLength)

    def forward(self, input):
        """
        Forward pass through the encoder.
        Args:
            input: The input tensor to encode (e.g., an image or feature map).

        Returns:
            category: The predicted category (classification output).
            res: A list of intermediate feature maps from different encoding stages.
        """
        # Apply the convolutional and residual blocks
        x1 = self.encodingBlock0(input)  # First encoding block
        x2 = self.encodingBlock1(x1)  # Second encoding block
        x3 = self.encodingBlock2(x2)  # Third encoding block
        x4 = self.encodingBlock3(x3)  # Fourth encoding block
        x5 = self.encodingBlock4(x4)  # Fifth encoding block
        x6 = self.encodingBlock5(x5)  # Sixth encoding block
        
        # Flatten the output from the last encoding block
        x7 = x6.view(x6.size(0), -1)  
        output = nn.functional.relu(x7)  # Apply ReLU activation

        # Pass the flattened features through the category classifier
        category = self.category(output)

        # Return the predicted category and the intermediate feature maps
        res = [x1, x2, x3, x4, x5, x6]
        return category, res