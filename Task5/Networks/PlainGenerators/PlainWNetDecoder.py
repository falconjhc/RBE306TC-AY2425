import torch
import torch.nn as nn

# Import the Deconvblock from the generator blocks
from Networks.PlainGenerators.PlainGeneratorBlocks import Deconvblock

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, normalization, out_channels=1, generator_dim=64):
        """
        Args:
            normalization: The type of normalization to apply (batchnorm, instancenorm, etc.).
            out_channels: Number of output channels (typically 1 for grayscale, 3 for RGB).
            generator_dim: The base number of feature maps for the generator (default: 64).
        """
        super(Decoder, self).__init__()
        self.normalization = normalization
        
        # Initialize deconvolutional blocks to progressively upsample the feature maps
        self.decodingBlock1 = Deconvblock(generator_dim * 8 * 2, generator_dim * 8, normalization=normalization)  # 512 channels
        self.decodingBlock2 = Deconvblock(generator_dim * 8 * 3, generator_dim * 8, normalization=normalization)  # 512 channels
        self.decodingBlock3 = Deconvblock(generator_dim * 8 * 3, generator_dim * 4, normalization=normalization)  # 256 channels
        self.decodingBlock4 = Deconvblock(generator_dim * 4 * 3, generator_dim * 2, normalization=normalization)  # 128 channels
        self.decodingBlock5 = Deconvblock(generator_dim * 2 * 2, generator_dim, normalization=normalization)  # 64 channels
        self.decodingBlock6 = Deconvblock(generator_dim * 2, out_channels, normalization='none', dropout=False)  # Output layer (1 channel by default)
    
    # Forward method to process the input through the decoder
    def forward(self, inputs_mix):
        """
        Args:
            inputs_mix: A list of tensors from the previous layers (usually from the encoder and mixer).
        
        Returns:
            A list of intermediate and final decoded outputs, including the generated output.
        """
        # Reverse the input list to process in the correct order
        input1, input2, input3, input4, input5, input6 = inputs_mix[::-1]

        # Pass the inputs through the deconvolutional layers step by step
        de_x1 = self.decodingBlock1(input1)

        # Concatenate the current output with the corresponding skip connection
        x1 = torch.concat((de_x1, input2), dim=1)
        de_x2 = self.decodingBlock2(x1)

        x2 = torch.concat((de_x2, input3), dim=1)
        de_x3 = self.decodingBlock3(x2)

        x3 = torch.concat((de_x3, input4), dim=1)
        de_x4 = self.decodingBlock4(x3)

        x4 = torch.concat((de_x4, input5), dim=1)
        de_x5 = self.decodingBlock5(x4)

        x5 = torch.concat((de_x5, input6), dim=1)
        de_x6 = self.decodingBlock6(x5)

        # Apply tanh activation to constrain the output between -1 and 1 (usually used for image generation)
        self.generated = torch.tanh(de_x6)

        # Return the decoded feature maps and the final generated image
        res = [de_x1, de_x2, de_x3, x4, de_x5, self.generated]
        return res