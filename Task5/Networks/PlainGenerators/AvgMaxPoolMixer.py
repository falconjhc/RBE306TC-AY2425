import torch
import torch.nn as nn
import numpy as np

# Importing a residual convolutional block from another module
from Networks.PlainGenerators.PlainGeneratorBlocks import ResidualConvBlock

# WNetMixer class designed to mix two sets of feature maps
class WNetMixer(nn.Module):
    def __init__(self, normalization, generator_dim=64):
        """
        Args:
            normalization (str): The type of normalization to use (e.g., batch normalization).
            generator_dim (int): The base dimension of the generator.
        """
        super(WNetMixer, self).__init__()
        
        # Initialize residual convolution blocks for mixing feature maps
        self.M = 5  # Define a constant for internal usage
        self.MixResidual1 = ResidualConvBlock(self.M-4, generator_dim, generator_dim, normalization=normalization)  # 64-channel block
        self.MixResidual2 = ResidualConvBlock(self.M-2, generator_dim*2, generator_dim*2, normalization=normalization)  # 128-channel block
        self.MixResidual3 = ResidualConvBlock(self.M, generator_dim*4, generator_dim*4, normalization=normalization)  # 256-channel block

    def forward(self, inputs1, inputs2):
        """
        Forward pass of the mixer that combines two sets of feature maps.
        Args:
            inputs1 (list): First set of feature maps.
            inputs2 (list): Second set of feature maps.
        Returns:
            list: The mixed feature maps.
        """
        # Unpack feature maps from inputs1 and inputs2
        input1_x1, input1_x2, input1_x3, input1_x4, input1_x5, input1_x6 = inputs1
        input2_x1, input2_x2, input2_x3, input2_x4, input2_x5, input2_x6 = inputs2

        # Apply the residual blocks to the feature maps from input1
        mix1 = self.MixResidual1(input1_x1)
        mix2 = self.MixResidual2(input1_x2)

        # Combine feature maps from input1 and input2
        mixer3 = torch.concat([self.MixResidual3(input1_x3), self.MixResidual3(input2_x3)], dim=1)
        mixer4 = torch.concat([input1_x4, input2_x4], dim=1)
        mixer5 = torch.concat([input1_x5, input2_x5], dim=1)
        mixer6 = torch.concat([input1_x6, input2_x6], dim=1)

        # Return the mixed feature maps as a list
        res = [mix1, mix2, mixer3, mixer4, mixer5, mixer6]

        return res
    
    # def reshape_tensor(self, input_tensor, is_train):
    #     """
    #     Reshape a tensor depending on whether it's training or validation.
    #     Args:
    #         input_tensor (torch.Tensor): The tensor to reshape.
    #         is_train (bool): Indicates if the operation is during training.
    #     Returns:
    #         torch.Tensor: The reshaped tensor.
    #     """
    #     # Select batch size based on training mode
    #     if is_train:
    #         batchsize = self.batchsize
    #     else:
    #         batchsize = self.val_batchsize

    #     # Reshape depending on tensor dimensions
    #     if len(input_tensor.shape) == 4:
    #         return input_tensor.reshape(batchsize, input_tensor.shape[0]//batchsize, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])
    #     elif len(input_tensor.shape) == 3:
    #         return input_tensor.reshape(batchsize, input_tensor.shape[0]//batchsize, input_tensor.shape[1], input_tensor.shape[2])
    #     elif len(input_tensor.shape) == 2:
    #         return input_tensor.reshape(batchsize, input_tensor.shape[0]//batchsize, input_tensor.shape[1])

# Example usage and testing
if __name__ == '__main__':
    # Define two sets of random input feature maps for testing
    inputs1 = [torch.randn(16, 64, 32, 32),
               torch.randn(16, 128, 16, 16),
               torch.randn(16, 256, 8, 8),
               torch.randn(16, 512, 4, 4),
               torch.randn(16, 512, 2, 2),
               torch.randn(16, 512, 1, 1)]
    inputs2 = [torch.randn(16, 64, 32, 32),
               torch.randn(16, 128, 16, 16),
               torch.randn(16, 256, 8, 8),
               torch.randn(16, 512, 4, 4),
               torch.randn(16, 512, 2, 2),
               torch.randn(16, 512, 1, 1)]

    # Create the WNetMixer and pass the inputs through it
    mixer = WNetMixer(normalization='batchnorm')  # Use batch normalization
    res = mixer(inputs1, inputs2)  # Get the mixed feature maps

    # Print the shapes of the output feature maps
    for x in res:
        print(x.shape)