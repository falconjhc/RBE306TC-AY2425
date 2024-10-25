import os
import random
import sys
import torch
import torchvision.transforms.functional as F

# Suppress warnings and TensorFlow32
torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
import cv2
from time import time
sys.path.append('./')
from Utilities.utils import set_random, cv2torch, read_file_to_dict
import time
from tqdm import tqdm
displayInterval = 25
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from Utilities.utils import PrintInfoLog


def RotationAugmentationToChannels(img, config, fill=1, style=False):
    """
    Applies random rotation and shear transformations to each channel of an image.
    
    Args:
        img (torch.Tensor): Input image tensor with shape (C, H, W).
        config (list): List containing rotation, shear, and translation parameters.
        fill (int): Value to use for padding.
        style (bool): If True, applies different rotation to each channel.
        
    Returns:
        torch.Tensor: Transformed image with shape (C, H, W).
    """
    rotation = config[0]
    shear = config[1]
    degrees = (-rotation, rotation)
    shear = (-shear, shear)
    angle = random.uniform(*degrees)
    shear = (random.uniform(-shear[0], shear[0]), random.uniform(-shear[1], shear[1]))

    transformed_channels = []  # List to store transformed channels
    
    for c in range(img.shape[0]):  # Apply transformation to each channel
        if style:  # Generate new random rotation for each style channel
            angle = random.uniform(*degrees)
            shear = (random.uniform(-shear[0], shear[0]), random.uniform(-shear[1], shear[1]))
        
        single_channel_img = img[c:c+1, :, :]  # Extract single channel (shape 1xHxW)

        # Apply affine transformation
        transformed_channel = F.affine(single_channel_img, angle=angle, translate=(0, 0),
                                       scale=1.0, shear=shear, fill=fill)

        transformed_channels.append(transformed_channel)

    # Stack all channels back together
    transformed_image = torch.cat(transformed_channels, dim=0)
    return transformed_image


def TranslationAugmentationToChannels(img, config, fill=1, style=False):
    """
    Applies random translation (shifting) transformations to each channel of an image.
    
    Args:
        img (torch.Tensor): Input image tensor with shape (C, H, W).
        config (list): List containing translation parameters.
        fill (int): Value to use for padding.
        style (bool): If True, applies different translation to each channel.
        
    Returns:
        torch.Tensor: Transformed image with shape (C, H, W).
    """
    max_translate = config[2]
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)

    transformed_channels = []  # List to store transformed channels
    
    for c in range(img.shape[0]):  # Apply translation to each channel
        if style:  # Generate new random translation for each style channel
            tx = random.randint(-max_translate, max_translate)
            ty = random.randint(-max_translate, max_translate)

        single_channel_img = img[c:c+1, :, :]  # Extract single channel (shape 1xHxW)

        # Pad the image to make room for the shift
        padded_img = F.pad(single_channel_img, (abs(tx), abs(ty), abs(tx), abs(ty)), fill=fill)

        # Get the original image size
        _, h, w = single_channel_img.shape

        # Crop the image back to original size
        cropped_img = F.crop(padded_img, abs(ty) if ty < 0 else 0, abs(tx) if tx < 0 else 0, h, w)
        transformed_channels.append(cropped_img)

    # Stack all channels back together
    transformed_image = torch.cat(transformed_channels, dim=0)
    return transformed_image


# Basic transformation for converting numpy arrays to PyTorch tensors
transformSingleContentGT = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy.ndarray to torch.Tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Rotation, shear, and translation configurations for different augmentation modes
rotationTranslationFull = [10, 10, 15]  # Full rotation, shear, and translation values
rotationTranslationHalf = [5, 5, 10]    # Half rotation, shear, and translation
rotationTranslationMinor = [3, 3, 5]    # Minor rotation, shear, and translation
rotationTranslationZero = [0, 0, 0]     # No rotation, shear, or translation

# Data augmentation transformations for combined content and style (Full augmentations)
transformFullCombinedContentGT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
    # Random resizing and cropping to (64, 64) with 75-100% scaling
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.75, 1.0), antialias=True),
])

transformFullStyle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
    # Random resizing and cropping for style images (no interpolation, just resize)
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.75, 1.0), antialias=True),
    transforms.ToTensor(),  # Converts to PyTorch tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Full transformations include both content + GT transformations and style transformations
fullTransformation = [transformFullCombinedContentGT, transformFullStyle, rotationTranslationFull]

# Data augmentation transformations for half-level augmentation
transformHalfCombinedContentGT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),  # 25% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.25),    # 25% chance of vertical flip
    # Random resizing and cropping with smaller scale range for less augmentation
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.85, 1.0), antialias=True),
])

transformHalfStyle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),  # 25% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.25),    # 25% chance of vertical flip
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.85, 1.0), antialias=True),
    transforms.ToTensor(),  # Converts to PyTorch tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Half transformations set with less aggressive augmentations
halfTransformation = [transformHalfCombinedContentGT, transformHalfStyle, rotationTranslationHalf]

# Minor data augmentation for minimal changes in images
transformMinorCombinedContentGT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),  # 10% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.1),    # 10% chance of vertical flip
    # Smaller resizing and cropping for minor augmentation
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.95, 1.0), antialias=True),
])

transformMinorStyle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),  # 10% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.1),    # 10% chance of vertical flip
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.95, 1.0), antialias=True),
    transforms.ToTensor(),  # Converts to PyTorch tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Minor transformations for content + style images
minorTransformation = [transformMinorCombinedContentGT, transformMinorStyle, rotationTranslationMinor]

# No augmentations applied, just basic conversions
transformZeroCombinedContentGT = transforms.Compose([])
transformZeroStyle = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy.ndarray to torch.Tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Zero transformations: No data augmentation
zeroTransformation = [transformZeroCombinedContentGT, transformZeroStyle, rotationTranslationZero]

# Defining augmentation dictionaries for different modes of training/testing
# Zero augmentation mode
transformTrainZero = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': zeroTransformation[0],
    'style': zeroTransformation[1],
    'rotationTranslation': zeroTransformation[2],
}

# Minor augmentation mode for training
transformTrainMinor = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': minorTransformation[0],
    'style': minorTransformation[1],
    'rotationTranslation': minorTransformation[2],
}

# Half augmentation mode for training
transformTrainHalf = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': halfTransformation[0],
    'style': halfTransformation[1],
    'rotationTranslation': halfTransformation[2],
}

# Full augmentation mode for training
transformTrainFull = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': fullTransformation[0],
    'style': fullTransformation[1],
    'rotationTranslation': fullTransformation[2],
}

# Test-time transformation (no augmentation, just tensor conversion)
transformTest = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': zeroTransformation[0],
    'style': zeroTransformation[1],
    'rotationTranslation': zeroTransformation[2],
}


class CharacterDataset(Dataset):
    def __init__(self, config, sessionLog, is_train=True):
        """
        Initialize the dataset by loading the content, ground truth, and style data from YAML files.
        
        Args:
            config (object): Configuration object containing dataset paths and parameters.
            sessionLog (object): Logging session to print logs.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """
        self.is_train = is_train
        self.sessionLog = sessionLog

        # Load paths to YAML files for train/test
        if is_train:
            self.gtYaml = os.path.join(config.datasetConfig.yamls, 'TrainGroundTruth.yaml')
            self.styleYaml = os.path.join(config.datasetConfig.yamls, 'TrainStyleReference.yaml')
        else:
            self.gtYaml = os.path.join(config.datasetConfig.yamls, 'TestGroundTruth.yaml')
            self.styleYaml = os.path.join(config.datasetConfig.yamls, 'TestStyleReference.yaml')

        self.contentYaml = os.path.join(config.datasetConfig.yamls, 'Content.yaml')

        # Load the number of content and style inputs
        self.input_content_num = config.datasetConfig.inputContentNum
        self.input_style_num = config.datasetConfig.inputStyleNum
        set_random()

        strat_time = time.time()  # Start timer to measure data loading time
        self.gtDataList = self.CreateDataList(self.gtYaml)  # Load the ground truth data

        # Load content and style YAML files
        with open(self.contentYaml, 'r', encoding='utf-8') as f:
            PrintInfoLog(self.sessionLog, "Loading " + self.contentYaml + '...', end='\r')
            contentFiles = yaml.load(f.read(), Loader=yaml.FullLoader)
            PrintInfoLog(self.sessionLog, "Loading " + self.contentYaml + ' completed.')

        with open(self.styleYaml, 'r', encoding='utf-8') as f:
            PrintInfoLog(self.sessionLog, "Loading " + self.styleYaml + '...', end='\r')
            styleFiles = yaml.load(f.read(), Loader=yaml.FullLoader)
            PrintInfoLog(self.sessionLog, "Loading " + self.styleYaml + ' completed.')

        self.contentList, self.styleList = [], []

        # Prepare content and style lists for each data point
        for idx, (_, label0, label1) in tqdm(enumerate(self.gtDataList), total=len(self.gtDataList), desc="Loading: "):
            contentFiles[label0] = [path for path in contentFiles[label0]]
            self.contentList.append(random.sample(contentFiles[label0], self.input_content_num))
            styleFiles[label1] = [path for path in styleFiles[label1]]
            self.styleList.append(random.sample(styleFiles[label1], self.input_style_num))

        # Initialize labels and one-hot encoding vectors
        self.label0order = config.datasetConfig.loadedLabel0Vec
                # One-hot encoding for content and style labels
        self.label1order = config.datasetConfig.loadedLabel1Vec
        self.onehotContent, self.onehotStyle = [0 for _ in range(len(self.label0order))], [0 for _ in range(len(self.label1order))]

        # Set data augmentation mode based on training or testing
        if self.is_train:
            self.ResetTrainAugment('NONE')
        else:
            self.augment = transformTest
        
        end_time = time.time()  # Measure the end time for data loading
        PrintInfoLog(self.sessionLog, f'dataset cost:{(end_time - strat_time):.2f}s')

    def __getitem__(self, index):
        """
        Retrieve a single item from the dataset by index.
        
        Args:
            index (int): Index of the item to retrieve.
        
        Returns:
            tuple: Containing content tensor, style tensor, ground truth tensor, one-hot encoded content label, and one-hot encoded style label.
        """
        # Load and process content tensors
        tensorContent = (torch.cat([cv2torch(content, self.augment['singleContentGT']) for content in self.contentList[index]], dim=0) - 0.5) * 2
        content = self.contentList[index][0][:-4].split('_')[-2]  # Extract content label from file path
        content = self.label0order[content]
        onehotContent = torch.tensor(self.onehotContent)
        onehotContent[content] = 1

        # Load and process ground truth tensor
        tensorGT = (cv2torch(self.gtDataList[index][0], self.augment['singleContentGT']) - 0.5) * 2
        gtAndContent = torch.cat((tensorContent, tensorGT), 0)

        # Apply rotation and translation augmentations
        gtAndContent = RotationAugmentationToChannels(gtAndContent, self.augment['rotationTranslation'])
        gtAndContent = TranslationAugmentationToChannels(gtAndContent, self.augment['rotationTranslation'])
        gtAndContent = self.augment['combinedContentGT'](gtAndContent)

        # Separate content and ground truth tensors
        tensorContent = gtAndContent[:-1, :, :]
        tensorGT = torch.unsqueeze(gtAndContent[-1, :, :], 0)

        # Load and process style tensors
        tensorStyle = (torch.cat([cv2torch(reference_style, self.augment['style']) for reference_style in self.styleList[index]], dim=0) - 0.5) * 2
        tensorStyle = RotationAugmentationToChannels(tensorStyle, self.augment['rotationTranslation'], style=True)
        tensorStyle = TranslationAugmentationToChannels(tensorStyle, self.augment['rotationTranslation'], style=True)

        # Extract style label
        style = self.styleList[index][0][:-4].split('_')[-1]
        while style[0] == '0' and len(style) > 1:
            style = style[1:]
        style = self.label1order[style]
        onehotStyle = torch.tensor(self.onehotStyle)
        onehotStyle[style] = 1

        return tensorContent.float(), tensorStyle.float(), tensorGT.float(), onehotContent.float(), onehotStyle.float()

    def __len__(self):
        """
        Get the total number of items in the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.gtDataList)

    def CreateDataList(self, yamlName):
        """
        Load ground truth data from a YAML file and create a list of data points.
        
        Args:
            yamlName (str): Path to the YAML file containing the data.
        
        Returns:
            list: A list of tuples containing paths, content labels, and style labels.
        """
        data_list = []
        with open(yamlName, 'r', encoding='utf-8') as f:
            PrintInfoLog(self.sessionLog, "Loading " + yamlName + '...', end='\r')
            iteration_files = yaml.load(f.read(), Loader=yaml.FullLoader)
            PrintInfoLog(self.sessionLog, "Loading " + yamlName + ' completed.')
        
        counter = 0
        for idx, (k, values) in tqdm(enumerate(iteration_files.items()), total=len(iteration_files.items()), desc="Test"):
            counter += 1
            path, label0, label1 = values
            data_list.append((path, label0, label1))
        return data_list

    def ResetTrainAugment(self, info):
        """
        Set the data augmentation mode for training.
        
        Args:
            info (str): Augmentation mode ('START', 'INITIAL', 'FULL', 'NONE').
        """
        if info == 'START':
            self.augment = transformTrainMinor
        elif info == 'INITIAL':
            self.augment = transformTrainHalf
        elif info == 'FULL':
            self.augment = transformTrainFull
        elif info == 'NONE':
            self.augment = None
        PrintInfoLog(self.sessionLog, "Switch to %s mode" % info)