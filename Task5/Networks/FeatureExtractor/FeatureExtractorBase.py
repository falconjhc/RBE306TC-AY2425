import sys
import torch
import torchvision.models as models  # Import pre-trained models from torchvision
import torch.nn as nn  # Import PyTorch's neural network module
sys.path.append('./')  # Add the current directory to the system path for module imports

# Dictionary mapping model names to corresponding torchvision models
featureExtractors = {
    'VGG11Net': models.vgg11_bn(),  # VGG11 with batch normalization
    'VGG13Net': models.vgg13_bn(),  # VGG13 with batch normalization
    'VGG16Net': models.vgg16_bn(),  # VGG16 with batch normalization
    'VGG19Net': models.vgg19_bn(),  # VGG19 with batch normalization
    'ResNet18': models.resnet18(),  # ResNet18
    'ResNet34': models.resnet34(),  # ResNet34
    'ResNet50': models.resnet50(),  # ResNet50
    'ResNet101': models.resnet101(),  # ResNet101
    'ResNet152': models.resnet152()   # ResNet152
}

# Define specific layers from VGG models to extract intermediate feature maps for evaluation
vggEvalLayers = {
    'VGG11Net': [3, 6, 13, 20, 27],  # Layers to extract for VGG11
    'VGG13Net': [5, 12, 19, 26, 33],  # Layers to extract for VGG13
    'VGG16Net': [5, 12, 22, 32, 42],  # Layers to extract for VGG16
    'VGG19Net': [5, 12, 25, 38, 51],  # Layers to extract for VGG19
}

# Base feature extractor class, determines if VGG or ResNet is used
class FeatureExtractorBase(nn.Module):
    def __init__(self, outputNums, modelSelect, type):
        super(FeatureExtractorBase, self).__init__()
        # Initialize VGG or ResNet model depending on modelSelect
        if 'VGG' in modelSelect:
            self.extractor = VGGNets(output_nums=outputNums, modelSelect=modelSelect, type=type)
        elif 'Res' in modelSelect:
            self.extractor = RESNets(output_nums=outputNums, modelSelect=modelSelect, type=type)

# VGG-based feature extractor class
class VGGNets(nn.Module):
    def __init__(self, output_nums, modelSelect, type):
        super(VGGNets, self).__init__()
        # Load the VGG model based on the selected architecture (VGG11, VGG13, etc.)
        self.model = featureExtractors[modelSelect]
        # Define which layers to use for feature extraction
        self.evalLayer = vggEvalLayers[modelSelect]
        
        # Custom classifier for 'content' type feature extraction
        self.model.classifierContent = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),  # 1x1 convolution
            nn.BatchNorm2d(1024),  # Batch normalization
            nn.ReLU(True),  # Activation
            nn.Flatten(1, 3),  # Flatten tensor
            nn.Linear(4096, output_nums)  # Fully connected layer for content classification
        )

        # Custom classifier for 'style' type feature extraction
        self.model.classifierStyle = nn.Sequential(
            nn.Flatten(1, 3),  # Flatten tensor
            nn.Linear(512 * 2 * 2, 1024),  # Fully connected layers for style classification
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, output_nums)
        )
        
        # Assign appropriate classifier based on the type ('content' or 'style')
        if type == 'content':
            self.model.classifierNew = self.model.classifierContent
        elif type == 'style':
            self.model.classifierNew = self.model.classifierStyle

    # Forward pass for the VGG-based feature extractor
    def forward(self, x):
        intermediate_outputs = []  # List to hold intermediate outputs for selected layers
        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel by repeating across the channel dimension
        
        # Extract feature maps from specified layers
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx in self.evalLayer:
                intermediate_outputs.append(x)  # Store intermediate output

        # Pass through the classifier for final prediction
        x = self.model.classifierNew(x)
        return x, intermediate_outputs  # Return both final output and intermediate features

# ResNet-based feature extractor class
class RESNets(nn.Module):
    def __init__(self, output_nums, modelSelect, type):
        super(RESNets, self).__init__()
        # Load the selected ResNet architecture
        self.model = featureExtractors[modelSelect]
        
        # Define the expansion factor based on the ResNet version
        if '18' in modelSelect or '34' in modelSelect:
            self.expansion = 1  # ResNet18 and ResNet34 have an expansion of 1
        elif '101' in modelSelect or '152' in modelSelect or '50' in modelSelect:
            self.expansion = 4  # ResNet50, ResNet101, and ResNet152 have an expansion of 4
        
        # Modify the final classifier to match the number of output classes
        self.model.classifier = nn.Linear(512 * self.expansion, output_nums)
        
    # Forward pass for the ResNet-based feature extractor
    def forward(self, x):
        intermediate_outputs = []  # List to hold intermediate outputs
        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel
        
        # Initial convolution and batch normalization
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        intermediate_outputs.append(x)  # Store the output of the first block
        
        # Continue passing through the ResNet layers
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        intermediate_outputs.append(x)
        
        x = self.model.layer2(x)
        intermediate_outputs.append(x)
        
        x = self.model.layer3(x)
        intermediate_outputs.append(x)
        
        x = self.model.layer4(x)
        intermediate_outputs.append(x)

        # Apply global average pooling and final classifier
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        
        x = self.model.classifier(x)  # Apply the classifier
        return x, intermediate_outputs  # Return final output and intermediate features