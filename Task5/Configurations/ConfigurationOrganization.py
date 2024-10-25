import os
import importlib
import torch
from Utilities.utils import SplitName, read_file_to_dict

# Network configuration object, defines the properties of a specific network
class NetworkConfigObject(object):
    def __init__(self, name, path, device='NA'):
        self.name = name  # Name of the network
        self.path = path  # Path to the network's weights or model file
        self.network = self.name.split('-')[0]  # Extract base network type from name
        self.device = device  # Target device (e.g., CPU or GPU)
        specifyDecoder = False
        
        # If the network involves decoders or encoders, extract their names
        if 'Decoder' in self.name:
            specifyDecoder = True
        
        if 'Encoder' in self.name or 'Mixer' in self.name or 'Decoder' in self.name:
            splits = self.name.split('-')
            for ii in splits:
                if 'Encoder' in ii:
                    self.encoder = ii
                    if not specifyDecoder:
                        self.decoder = ii
                if 'Mixer' in ii:
                    self.mixer = ii
                if 'Decoder' in ii:
                    self.decoder = ii
                    
        # Ensure decoder is appropriately named based on the encoder
        if hasattr(self, 'decoder'):
            self.decoder = self.decoder.replace('Encoder', 'Decoder')
            if not specifyDecoder:
                splits = SplitName(self.decoder)
                decoderNew = splits[0]
                for ii in range(len(splits) - 1):
                    decoderNew = decoderNew + splits[-ii-1]
                self.decoder = decoderNew


# Configuration object for the dataset
class DatasetConfigObject(object):
    def __init__(self, augmentation, inputStyleNum, inputContentNum, imgWidth, channels, label0VecTxt, label1VecTxt, dataRoot):
        self.augmentation = augmentation  # Type of data augmentation to apply
        self.inputStyleNum = inputStyleNum  # Number of input style images
        self.inputContentNum = inputContentNum  # Number of input content images
        self.imgWidth = imgWidth  # Width of input images
        self.channels = channels  # Number of channels in the images (e.g., RGB or grayscale)
        # Load label vectors from the given files
        self.loadedLabel0Vec = read_file_to_dict(label0VecTxt)
        self.loadedLabel1Vec = read_file_to_dict(label1VecTxt)


# User interface settings for experiment directories, logging, and image output
class UserInterfaceObj(object):
    def __init__(self, expID, expDir, logDir, resumeTrain, imgDir, skipTest):
        self.expID = expID  # Experiment ID
        self.expDir = expDir  # Directory to store experiment results
        self.logDir = logDir  # Directory to store logs
        self.resumeTrain = resumeTrain  # Whether to resume training from a checkpoint
        self.skipTest = skipTest  # Whether to skip testing phase
        self.trainImageDir = imgDir  # Directory to store training images


# Training parameters such as optimizer type, batch size, learning rate, etc.
class TrainParamObject(object):
    def __init__(self, args, seed, optimizer, gradientNorm, initTrainEpochs, debugMode):
        self.optimizer = optimizer  # Type of optimizer to use (e.g., Adam, SGD)
        self.initTrainEpochs = initTrainEpochs  # Number of initial training epochs
        self.gradientNorm = gradientNorm  # Whether to use gradient normalization
        self.epochs = args.epochs  # Total number of training epochs
        self.batchSize = args.batchSize  # Batch size for training
        self.initLr = args.initLr  # Initial learning rate
        self.debugMode = debugMode  # Whether to run in debug mode
        self.seed = seed  # Seed for random number generators to ensure reproducibility


# Main parameter setting object that handles experiment configuration
class ParameterSetting(object):
    def __init__(self, config, args):
        self.config = config
        self.config.debug = args.debug  # Whether debug mode is enabled

        # Identify available CPU and GPU devices
        avalialbe_cpu, available_gpu = self.FindAvailableDevices()
        # Select device (CPU or GPU) based on available hardware and user input
        self.config.device = self.CheckGPUs(args.device, available_gpu, avalialbe_cpu)

        # Set the WNet (Wide Network) generator model
        self.config.wnet = args.wnet

        # Generate a unique experiment ID based on WNet, encoder, mixer, and decoder
        self.config.expID = self.GenerateExpID(self.config.wnet, args.encoder, args.mixer, args.decoder)
        
        # Define directories for model checkpoints, logs, and training images
        self.config.trainModelDir = os.path.join(self.config.expDir, 'Models', self.config.expID)
        self.config.trainLogDir = os.path.join(self.config.expDir, 'Logs', self.config.expID)
        self.config.trainImageDir = os.path.join(self.config.expDir, 'Images', self.config.expID)
        
        # Initialize user interface settings
        self.config.userInterface = UserInterfaceObj(expID=self.config.expID, 
                                                     expDir=self.config.trainModelDir, 
                                                     logDir=self.config.trainLogDir, 
                                                     imgDir=self.config.trainImageDir,
                                                     resumeTrain=args.resumeTrain,
                                                     skipTest=args.skipTest)

        # Remove unnecessary config attributes after setting them
        self.config.pop('expDir', None)
        self.config.pop('trainModelDir', None)
        self.config.pop('trainLogDir', None)
        self.config.pop('skipTest', None)

        # Set generator and discriminator networks
        self.config.generator = NetworkConfigObject(name=self.config.expID,
                                                    path=os.path.join(self.config.userInterface.expDir, 'Generator'))
        self.config.discriminator = NetworkConfigObject(name=self.config.discriminator, 
                                                        path=os.path.join(self.config.userInterface.expDir, 'Discriminator'))

        # Configure feature extractors if paths are specified
        if self.config.TrueFakeExtractorPath:
            self.config.extractorTrueFake = NetworkConfigObject(name='TrueFakeFeatureExtractor', 
                                                                path=self.config.TrueFakeExtractorPath)

        if self.config.ContentExtractorPath:
            self.config.extractorContent = self.ProcessNetworks(self.config.ContentExtractorPath)
        
        if self.config.StyleExtractorPath:
            self.config.extractorStyle = self.ProcessNetworks(self.config.StyleExtractorPath)

        # Remove unnecessary attributes related to feature extraction
        self.config.pop('featureExtractorDevice', None)
        self.config.pop('TrueFakeExtractorPath', None)
        self.config.pop('ContentExtractorPath', None)
        self.config.pop('StyleExtractorPath', None)

        # Load dataset configuration and paths
        dataRootPath = importlib.import_module('.' + args.config, package='Configurations').dataPathRoot
        self.config.datasetConfig = DatasetConfigObject(augmentation=self.config.augmentation,
                                                        inputStyleNum=self.config.inputStyleNum, 
                                                        inputContentNum=self.config.inputContentNum,
                                                        imgWidth=self.config.imgWidth,
                                                        channels=self.config.channels,
                                                        dataRoot=dataRootPath,
                                                        label1VecTxt=self.config.FullLabel1Vec,
                                                        label0VecTxt=self.config.FullLabel0Vec)
        self.config.datasetConfig.yamls = self.config.YamlPackage
        
        # Remove unnecessary dataset configuration attributes
        self.config.pop('YamlPackage', None)
        self.config.pop('inputStyleNum', None)
        self.config.pop('imgWidth', None)
        self.config.pop('channels', None)
        self.config.pop('label1VecTxt', None)
        self.config.pop('label0VecTxt', None)

        # Set training parameters
        self.config.trainParams = TrainParamObject(args=args, 
                                                   seed=self.config.seed,
                                                   optimizer=self.config.optimizer, 
                                                   gradientNorm=self.config.gradNorm,
                                                   initTrainEpochs=self.config.initTrainEpochs,
                                                   debugMode=self.config.debugMode)
        
        # Remove unnecessary training parameter attributes
        self.config.pop('optimizer', None)
        self.config.pop('initTrainEpochs', None)
        self.config.pop('debugMode', None)
        self.config.pop('seed', None)
        self.config.pop('gradNorm', None)

    # Method to process a list of network paths and create NetworkConfigObjects
    def ProcessNetworks(self, models):
        output = []
        for thisPath in models:
            thisName = thisPath.split('/')[-2]
            this_network_obj = NetworkConfigObject(name=thisName, path=thisPath)
            output.append(this_network_obj)
        return output

    # Method to check the availability of selected GPUs or fallback to CPU
    def CheckGPUs(self, selectedDevice, availableGPU, availableCPU):
        selectedDeviceList = []
        if 'cpu' not in selectedDevice:
            for ii in selectedDevice:
                if str.isdigit(ii) and 'cuda:' + ii in availableGPU:
                    selectedDeviceList.append('cuda:' + ii)
                elif not 'cuda:' + ii in availableGPU and str.isdigit(ii):
                    print("Device ERROR: GPU not available.")
        else:
            selectedDeviceList = availableCPU
        return selectedDeviceList

    # Method to find available CPU and GPU devices
    def FindAvailableDevices(self):
        # Get CPU device
        cpu_device = ['cpu']

        # Get GPU devices if available
        gpu_device = []
        if torch.cuda.is_available():
            gpu_device = ['cuda:' + str(i) for i in range(torch.cuda.device_count())]

        # Convert devices to string and sort
        cpu_device = [str(x) for x in cpu_device]
        gpu_device = [str(x) for x in gpu_device]

        # Print available devices
        print("Available CPU: %s with number: %d" % (cpu_device, len(cpu_device)))
        print("Available GPU: %s with number: %d" % (gpu_device, len(gpu_device)))

        # Sort device lists
        cpu_device.sort()
        gpu_device.sort()

        return cpu_device, gpu_device

    # Method to generate a unique experiment ID
    def GenerateExpID(self, wnet, encoder, mixer, decoder):
        if wnet == 'general':
            id = "ExpGeneralized-%s-%s-%s" % (self.config.expID, encoder, mixer)
            if decoder is not None:
                id = id + "-%s" % decoder
        elif wnet == 'plain':
            id = "ExpPlain-%s" % self.config.expID
        if not self.config.discriminator == 'NA':
            id = id + "-%s" % self.config.discriminator
        return id