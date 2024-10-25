import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings
import torch
import torch.nn.functional as F
import torch.nn as nn

# Define constants for the multi-layer perceptron (MLP) ratio and patch size
mlpRatio=4
patchSize=4


import torch
import torch.nn.functional as F
from Networks.GeneralizedGenerator.VisionTransformer import VisionTransformer as VIT
from Networks.GeneralizedGenerator.VisionTransformer import PatchMerging as PatchMerger
from Networks.GeneralizedGenerator.VisionTransformer import PatchExpansion as PatchExpander

import torch.nn as nn

# Set bias for convolutional neural networks (CNN)
cnnBias=True

# Define LeakyReLU activation with a negative slope of 0.2
RELU = nn.LeakyReLU(0.2, inplace=True)

# 1x1 transposed convolution (deconvolution) with stride and output padding for exact size
def deconv1x1(in_planes, out_planes, stride=1,cnnBias=False):
    """1x1 transposed convolution (deconvolution) with stride and output padding for exact size"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, 
                              output_padding=stride-1, bias=cnnBias)

# 3x3 transposed convolution (deconvolution) to double the spatial size
def deconv3x3(in_planes, out_planes, stride=1,cnnBias=False):
    """3x3 transposed convolution (deconvolution) to double the spatial size"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, 
                              output_padding=stride-1, bias=cnnBias)
    
# 3x3 convolution with padding, commonly used in CNN layers
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,cnnBias=False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=cnnBias,
        dilation=dilation,
    )

# 1x1 convolution, often used for dimensionality reduction
def conv1x1(in_planes: int, out_planes: int, stride: int = 1,cnnBias=False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=cnnBias)
    
# Class to represent the features extracted by convolutional and vision transformer blocks
class BlockFeature(object):
    def __init__(self, cnn, vit=None):
        self.cnn = cnn
        
        # If Vision Transformer features are not provided, generate them using unfolding patches from the CNN output
        if vit is not None:
            self.vit = vit
        elif cnn.shape[2] >= patchSize:  # Height is at index 2 for PyTorch tensors (batch, channels, height, width)
            # Use unfold to extract patches in PyTorch
            vit = cnn.unfold(2, patchSize, patchSize).unfold(3, patchSize, patchSize)
            
            # Extract patchW and patchH based on the unfolded dimensions
            patchW = vit.size(2)  # Patch width # Equivalent to vit.shape[1] in TensorFlow 
            patchH = vit.size(3)  # Patch height # Equivalent to vit.shape[2] in TensorFlow
            
            # Reshape vit to match the expected shape
            self.vit = vit.contiguous().view(vit.size(0), patchW * patchH, patchSize * patchSize * cnn.size(1))
        else:
            self.vit = None
    
    # Method to return shapes of the CNN and Vision Transformer features as a list of strings
    def ProcessOutputToList(self):
        if self.vit is not None:
            return [str(self.cnn.shape)[11:-1], str(self.vit.shape)[11:-1]]
        else:
            return [str(self.cnn.shape)[11:-1], 'None']
            
# Class for extracting patches from an input feature map
class PatchExtractor(nn.Module):
    def __init__(self, featureDim, patchSize=4):
        super(PatchExtractor, self).__init__()
        self.patchSize=patchSize
        self.featureDim=featureDim
        self.patchDim = self.patchSize*self.patchSize*self.featureDim
        
    def forward(self, x):
        batch_size = x.size(0)  # Get batch size from images tensor
        patches = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        patches = patches.contiguous().view(batch_size, -1, self.patchSize * self.patchSize * x.size(1))  # Reshape the patches        
        return patches
    
# Encoding block for bottleneck layers, used in architectures like ResNet
class EncodingBottleneckBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(EncodingBottleneckBlock,self).__init__()
        
        # Determine stride based on input and output dimensions or configuration for downsampling
        if inDims['HW']==outDims['HW']*2 or config['downsample']:
            stride=2
        else:
            stride=1
        
        # Set up 1x1 convolutions and batch normalization for the identity shortcut
        self.conv_identity = conv1x1(inDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        # Define the main convolutions for the bottleneck block
        self.conv1 = conv1x1(inDims['MapC'], outDims['MapC'],cnnBias=cnnBias)
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        self.conv2 = conv3x3(outDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        self.conv3 = conv1x1(outDims['MapC'], outDims['MapC'],cnnBias=cnnBias)
        self.bn3 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = RELU
        self.stride = stride
        
        # Patch extractor for Vision Transformer integration
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
            
        
    def forward(self, x):
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)

        # Apply convolutions and ReLU activations
        outcnn = self.conv1(x.cnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv3(outcnn)
        outcnn = self.bn3(outcnn)
        
        # Add the identity shortcut
        outcnn += identity
        outcnn = self.relu(outcnn)
        
        # Extract patches for Vision Transformer
        outvit = self.patchExtractor(outcnn)
        out  =BlockFeature(cnn=outcnn, vit=outvit)
        return out

# Similar to the bottleneck block, but for basic residual connections  
class EncodingBasicBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(EncodingBasicBlock,self).__init__()
        if inDims['HW']==outDims['HW']*2 or config['downsample']:
            stride=2
        else:
            stride=1
            
        # Define the identity shortcut
        self.conv_identity = conv3x3(inDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        # Define the main convolutions for the basic block
        self.conv1 = conv3x3(inDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = RELU
        self.conv2 = conv3x3(outDims['MapC'], outDims['MapC'],cnnBias=cnnBias)
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        self.stride = stride
        
        # Patch extractor for Vision Transformer integration
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
    
    def forward(self,x):
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)

        # Apply convolutions and ReLU activations
        outcnn = self.conv1(x.cnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)

        # Add the identity shortcut
        outcnn += identity
        outcnn = self.relu(outcnn)
        
        # Extract patches for Vision Transformer
        outvit = self.patchExtractor(outcnn)
        out =BlockFeature(cnn=outcnn, vit=outvit)
        return out
    
# Encoding block for Vision Transformer integration
class EncodingVisionTransformerBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(EncodingVisionTransformerBlock,self).__init__()
        
        
        self.downsample = False
        self.inDims=inDims
        self.outDims=outDims
        if inDims['VitDim'] == outDims['VitDim'] * 4 or config['downsample']:
            self.downsample = True

        # Parse configuration for Vision Transformer
        _, numVit, numHead = config['option'].split("@")
        numVit = int(numVit)
        numHead = int(numHead)
        
        # Initialize Vision Transformer and Patch Merging
        self.vit = VIT(image_size=inDims['HW'],
                       patch_size=patchSize,
                       num_layers=numVit, num_heads=numHead,
                       d_model=outDims['VitC'],
                       mlp_dim=outDims['VitDim']*mlpRatio,
                       patchDim = inDims['PatchDim'])
        self.merger = PatchMerger(dim=outDims['VitC'])
        
    def forward(self,x):
        # Apply Vision Transformer (ViT) to the input feature map
        outvit = self.vit(x.vit)
        
        # Downsample the output if necessary
        if self.downsample:
            outvit=self.merger(outvit)
        
        # Calculate CNN representation if the output dimensions match
        count=1
        for ii in range(len(outvit.shape)-1): count = count * outvit.shape[ii+1]
        if count%(self.outDims['HW']*self.outDims['HW'])==0:
            xcnn = outvit.view(outvit.size(0), -1, self.outDims['HW'],self.outDims['HW'])
        else:
            xcnn=None
        return BlockFeature(cnn=xcnn, vit=outvit)
        
        
# Decoding block for Vision Transformer integration
class DecodingVisionTransformerBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(DecodingVisionTransformerBlock,self).__init__()
        
        
        self.upsample = False
        self.inDims=inDims
        self.outDims=outDims
        if inDims['VitDim'] == outDims['VitDim'] // 4 or config['upsample']:
            self.upsample = True
            
        # Parse the config string to determine the number of layers and heads in the ViT
        _, numVit, numHead = config['option'].split("@")
        numVit = int(numVit)
        numHead = int(numHead)
        
        # Initialize the Vision Transformer and Patch Expander for upsampling
        self.vit = VIT(image_size=outDims['HW'],
                       patch_size=patchSize,
                       num_layers=numVit, num_heads=numHead,
                       d_model=outDims['VitC'],
                       mlp_dim=outDims['VitDim']*mlpRatio,
                       patchDim = inDims['PatchDim'])
        self.expander = PatchExpander(dim=outDims['VitDim'])
        
    def forward(self,x, enc=None):
        # Concatenate the encoder features with the input if provided
        if enc is not None:
            outvit = torch.concat((x.vit, enc.vit), dim=-1)
        else:
            outvit = x.vit 
        
        # Perform upsampling if required
        if self.upsample:
            outvit = self.expander(outvit)
        else:
            outvit=x.vit
        
        
        # Apply the Vision Transformer
        outvit = self.vit(outvit)
            
        # Reshape the ViT output to a CNN-like feature map
        xcnn = outvit.view(outvit.size(0), -1, self.outDims['HW'],self.outDims['HW'])
        return BlockFeature(cnn=xcnn, vit=outvit)
    
    
# Decoding block for basic residual layers 
class DecodingBasicBlock(nn.Module):
    def __init__(self, inDims, outDims, config):
        super(DecodingBasicBlock,self).__init__()
        self.lastLayer = config['lastLayer']
        if inDims['HW']==outDims['HW']//2 or config['upsample']:
            stride=2
        else:
            stride=1
            
        # Define the identity shortcut
        self.conv_identity = deconv3x3(inDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        # If skip connections are present, concatenate them with the input
        if config['skip']  is not None:
            self.conv1 = deconv3x3(inDims['MapC']+config['skip'][0], outDims['MapC'], stride,cnnBias=cnnBias)
        else:
            self.conv1 = deconv3x3(inDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = RELU
        self.tanh = nn.Tanh()
        self.conv2 = deconv3x3(outDims['MapC'], outDims['MapC'],cnnBias=cnnBias)
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        self.stride = stride
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
    
    def forward(self, x, enc=None):
        # Apply identity shortcut
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)
        
        # Concatenate encoder features if available
        if enc is not None:
            outcnn = torch.concat((x.cnn, enc.cnn), dim=1)
        else:
            outcnn=x.cnn
        
        # Perform the first and second convolutions
        outcnn = self.conv1(outcnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)
        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)

        # Add the identity shortcut and apply activation functions
        outcnn += identity
        if not self.lastLayer:
            outcnn = self.relu(outcnn)
        else:
            outcnn = self.tanh(outcnn)
        
        # Extract patches for Vision Transformer
        outvit = self.patchExtractor(outcnn)
        out =BlockFeature(cnn=outcnn, vit=outvit)
        return out

# Decoding block for bottleneck layers
class DecodingBottleneckBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(DecodingBottleneckBlock,self).__init__()
        self.lastLayer = config['lastLayer']
        if inDims['HW']==outDims['HW']//2 or config['upsample']:
            stride=2
        else:
            stride=1
        
        # Define the identity shortcut
        self.conv_identity = deconv1x1(inDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        # First convolution
        self.conv1 = deconv1x1(inDims['MapC'], outDims['MapC'],cnnBias=cnnBias)
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        
        
        # If skip connections are present, concatenate them with the input
        if config['skip'] is not None:
            self.conv2 = deconv3x3(outDims['MapC']+config['skip'][0], outDims['MapC'], stride,cnnBias=cnnBias)
        else:
            self.conv2 = deconv3x3(outDims['MapC'], outDims['MapC'], stride,cnnBias=cnnBias)
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        
        # Third convolution
        self.conv3 = deconv1x1(outDims['MapC'], outDims['MapC'],cnnBias=cnnBias)
        self.bn3 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = RELU
        self.tanh = nn.Tanh()
        self.stride = stride
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
            
        
    def forward(self, x, enc=None):
        # Apply identity shortcut
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)

        # Perform the first convolution
        outcnn = self.conv1(x.cnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)

        # Concatenate encoder features if available
        if enc is not None:
            outcnn = torch.concat((outcnn, enc.cnn), dim=1)

        # Perform the second convolution    
        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)
        outcnn = self.relu(outcnn)

        # Perform the third convolution and add the identity shortcut
        outcnn = self.conv3(outcnn)
        outcnn = self.bn3(outcnn)
        
        outcnn += identity
        if not self.lastLayer:
            outcnn = self.relu(outcnn)
        else:
            outcnn = self.tanh(outcnn)
        
        # Extract patches for Vision Transformer
        outvit = self.patchExtractor(outcnn)
        out  =BlockFeature(cnn=outcnn, vit=outvit)
        return out
    