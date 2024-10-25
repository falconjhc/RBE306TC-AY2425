import sys
import torch
torch.set_warn_always(False)  # Suppress PyTorch warnings
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)  # Ignore future warnings
import shutup
shutup.please()  # Silence other library warnings

import torch.nn as nn

# Add project directories to the system path
sys.path.append('./')

# Import components for the WNet architecture
from Networks.PlainGenerators.AvgMaxPoolMixer import WNetMixer
from Networks.PlainGenerators.PlainWNetDecoder import Decoder
from Networks.PlainGenerators.PlainWNetEncoder import Encoder
from Utilities.utils import set_random, PrintInfoLog

# Define dimensions for CNN and Vision Transformer (ViT) components
cnnDim = 64
vitDim = 96

# Define the WNet generator class, which consists of encoder, mixer, and decoder components
class WNetGenerator(nn.Module):
    def __init__(self, config, sessionLog):
        """
        Args:
            config: Configuration object for the network.
            sessionLog: Logger for recording messages during training.
        """
        super(WNetGenerator, self).__init__()    

        self.sessionLog = sessionLog  # Logger for the session
        
        # Set the training mode flag and config
        self.is_train = True
        self.config = config
        set_random()  # Set random seed for reproducibility

        # Initialize content encoder with specific input channels and category length
        self.contentEncoder = Encoder(input_channels=self.config.datasetConfig.channels * self.config.datasetConfig.inputContentNum,
                                      loadedCategoryLength=len(config.datasetConfig.loadedLabel0Vec),
                                      generator_dim=cnnDim, normalization='batchnorm')
        self.contentEncoder.train()  # Set to training mode
        self.contentEncoder.cuda()  # Move the model to GPU
        
        # Initialize style encoder with specific input channels and category length
        self.styleEncoder = Encoder(input_channels=self.config.datasetConfig.channels,
                                    loadedCategoryLength=len(config.datasetConfig.loadedLabel1Vec),
                                    generator_dim=cnnDim, normalization='batchnorm')
        self.styleEncoder.train()  # Set to training mode
        self.styleEncoder.cuda()  # Move the model to GPU
        
        # Initialize the mixer block to combine content and style features
        self.mixer = WNetMixer(generator_dim=cnnDim, normalization='batchnorm')
        self.mixer.train()  # Set to training mode
        self.mixer.cuda()  # Move the model to GPU

        # Initialize the decoder for generating images from the mixed features
        self.decoder = Decoder(out_channels=self.config.datasetConfig.channels, generator_dim=cnnDim, normalization='batchnorm')
        self.decoder.train()  # Set to training mode
        self.decoder.cuda()  # Move the model to GPU
    
    def forward(self, content_inputs, style_inputs, GT, is_train=True):
        """
        Forward pass through the network.

        Args:
            content_inputs: Input tensor for content images.
            style_inputs: Input tensor for style images.
            GT: Ground truth tensor for comparison.
            is_train: Whether the network is in training mode.
        
        Returns:
            Encoded content features, style features, categories, and the generated image.
        """
        # Encode content and style inputs
        content_category_onReal, content_outputs_onReal = self.contentEncoder(content_inputs)
        style_category_onReal, style_outputs_onReal = self.styleEncoder(style_inputs)
        enc_content_list_onReal, enc_style_list_onReal = [], [] 
        reshaped_style_list_onReal = []
        
        # Process each output from the content and style encoders
        for content, style in zip(content_outputs_onReal, style_outputs_onReal):
            reshaped_style_outputs = self.TensorReshape(style, is_train)  # Reshape style features for batch processing
            max_style_output = torch.max(reshaped_style_outputs, dim=1)[0]  # Max-pooling on style outputs
            enc_content_list_onReal.append(content)
            enc_style_list_onReal.append(max_style_output)
            reshaped_style_list_onReal.append(reshaped_style_outputs)

        # Mix the encoded content and style features
        mix_output = self.mixer(enc_content_list_onReal, enc_style_list_onReal)
        
        # Decode the mixed features to generate the final output image
        decode_output_list = self.decoder(mix_output)
        generated = decode_output_list[-1]

        # Encode the ground truth image
        GT_content_category, GT_content_outputs = self.contentEncoder(GT.repeat(1, GT.shape[1] * self.config.datasetConfig.inputContentNum, 1, 1))
        GT_style_category, GT_style_outputs = self.styleEncoder(GT)
        
        # Encode the generated image
        contentCategoryOnGenerated, contentFeaturesOnGenerated = self.contentEncoder(generated.repeat((1, self.config.datasetConfig.inputContentNum, 1, 1)))
        styleCategoryOnGenerated, styleFeaturesOnGenerated = self.styleEncoder(generated)

        # Max-pooling operations to extract category features
        max_content_category_onReal = torch.max(self.TensorReshape(content_category_onReal, is_train), dim=1)[0]
        max_style_category_onReal = torch.max(self.TensorReshape(style_category_onReal, is_train), dim=1)[0]
        max_lossCategoryContentFakeerated = torch.max(self.TensorReshape(contentCategoryOnGenerated, is_train), dim=1)[0]
        max_lossCategoryStyleFakeerated = torch.max(self.TensorReshape(styleCategoryOnGenerated, is_train), dim=1)[0]

        # Dictionary to store encoded features and categories for content and style
        encodedContentFeatures = {}
        encodedStyleFeatures = {}
        encodedContentCategory = {}
        encodedStyleCategory = {}
        
        # Store real, fake, and ground truth features for content
        encodedContentFeatures.update({'real': enc_content_list_onReal[-1]})
        encodedContentFeatures.update({'fake': contentFeaturesOnGenerated[-1]})
        encodedContentFeatures.update({'groundtruth': GT_content_outputs[-1]})
        
        # Store real, fake, and ground truth features for style
        encodedStyleFeatures.update({'real': reshaped_style_list_onReal[-1]})
        encodedStyleFeatures.update({'fake': styleFeaturesOnGenerated[-1]})
        encodedStyleFeatures.update({'groundtruth': GT_style_outputs[-1]})
        
        # Store content and style categories
        encodedContentCategory.update({'real': max_content_category_onReal})
        encodedContentCategory.update({'fake': max_lossCategoryContentFakeerated})
        encodedContentCategory.update({'groundtruth': GT_content_category})
        
        encodedStyleCategory.update({'real': max_style_category_onReal})
        encodedStyleCategory.update({'fake': max_lossCategoryStyleFakeerated})
        encodedStyleCategory.update({'groundtruth': GT_style_category})
        
        return encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated
    
    # Method to reshape tensors based on training mode and batch size
    def TensorReshape(self, input_tensor, is_train):
        """
        Reshape input tensor based on the training mode and batch size.

        Args:
            input_tensor: Tensor to be reshaped.
            is_train: Boolean flag for training mode.
        
        Returns:
            Reshaped tensor.
        """
        if len(input_tensor.shape) == 4:
            return input_tensor.reshape(self.config.trainParams.batchSize, input_tensor.shape[0] // self.config.trainParams.batchSize, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])
        elif len(input_tensor.shape) == 3:
            return input_tensor.reshape(self.config.trainParams.batchSize, input_tensor.shape[0] // self.config.trainParams.batchSize, input_tensor.shape[1], input_tensor.shape[2])
        elif len(input_tensor.shape) == 2:
            return input_tensor.reshape(self.config.trainParams.batchSize, input_tensor.shape[0] // self.config.trainParams.batchSize, input_tensor.shape[1])