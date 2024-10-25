# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

# System and path setup
import sys
sys.path.append('../')
sys.path.append('../../')

# Required imports for environment and image processing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
import torch.nn as nn
import numpy as np 

# Importing custom utility functions
from Utilities.utils import SplitName, FindKeys
from Networks.GeneralizedGenerator.GeneralizedGeneratorBlocks import EncodingBottleneckBlock as BottleneckBlock
from Networks.GeneralizedGenerator.GeneralizedGeneratorBlocks import BlockFeature
from Networks.GeneralizedGenerator.GeneralizedGeneratorBlocks import EncodingVisionTransformerBlock as VisionTransformerBlock
from Networks.GeneralizedGenerator.GeneralizedGeneratorBlocks import patchSize


# Constants
eps = 1e-9
cnnDim=64
vitDim=96

# Dictionary for block encoding methods
BlockEncDict={'Cv': BottleneckBlock,
           'Cbb': BottleneckBlock,
           'Cbn': BottleneckBlock,
           'Vit': VisionTransformerBlock}

# Dictionary for style fusing methods
StyleFusingDict={'Max': torch.max,
                'Min': torch.min,
                'Avg': torch.mean}

# Separator for printing
print_separater="#########################################################"

# Define WNetMixer class for mixing encoded features
class WNetMixer(nn.Module):
    def __init__(self,  encodingFeatureShape, config):
        super(WNetMixer, self).__init__()
        self.config=config
        self.encodingFeatureShape=encodingFeatureShape
        _, self.styleFusingMethod, fusionContentStyle  = SplitName(config.generator.mixer)[:3]
        self.architectureEncoderList = SplitName(self.config.generator.encoder)[1:]
        self.architectureDecoderList = SplitName(self.config.generator.decoder)[1:]
        
        # Determine fusion style: Residual, Dense, or Simple
        if 'Res' in fusionContentStyle: # Residual
            self.fusionContentStyle='Res'
        elif 'Dns' in fusionContentStyle: # Dense
            self.fusionContentStyle='Dns'
        elif 'Smp' in fusionContentStyle:
            self.fusionContentStyle='Smp' # Simple Connection
        
        if self.fusionContentStyle=='Dns' or self.fusionContentStyle=='Res':
            residualBlockNum, residualAtLayer = fusionContentStyle[len(self.fusionContentStyle):].split('@')
            self.residualBlockNum=int(residualBlockNum)
            self.residualAtLayer=int(residualAtLayer)
        
        # Define the fusing operations
        self.fusingOperations = []
        if self.fusionContentStyle != 'Smp': # not simple mixer
            for ii in range(len(self.encodingFeatureShape)):
                if ii ==0:
                    continue # skip the STEM feature
                thisFeatureCNNShape = self.encodingFeatureShape[ii][0][1:-1].split(',')
                thisFeatureVitShape = self.encodingFeatureShape[ii][1][1:-1].split(',')
                thisArchitecture = self.architectureEncoderList[ii-1]
                    
                
                if ii <self.residualAtLayer:
                    thisResidualNum = max(self.residualBlockNum-ii*2,1)
                    
                    # Generate channel lists for CNN and ViT
                    channelListCNN = [int(jj) for jj in np.linspace(int(thisFeatureCNNShape[1]),
                                                                    int(thisFeatureCNNShape[1])//2,
                                                                    thisResidualNum+1).tolist()]
                    channelListVit = [int(jj) for jj in np.linspace(int(thisFeatureVitShape[-1]),
                                                                    int(thisFeatureVitShape[-1])//2,
                                                                    thisResidualNum+1).tolist()]
                    channelListCNN = [jj*2 for jj in channelListCNN] 
                    channelListVit = [jj*2 for jj in channelListVit] 
                    thisBlock = FindKeys(BlockEncDict, thisArchitecture)[0]
                    thisCnnHW=int(thisFeatureCNNShape[2])
                    thisVitDim = int(thisFeatureVitShape[1])
                    
                    
                    
                    # if 'C' in thisArchitecture:
                        
                    
                    thisResList=[]
                    # thisResidualNum=1
                    for jj in range(thisResidualNum):
                        
                        thisPatchDim=-1
                        if "Vit" in thisArchitecture:
                            thisPatchDim = channelListVit[jj]
                            thisPatchDim= thisPatchDim-thisPatchDim%int(thisArchitecture.split('@')[-1])
                        thisResult = thisBlock(inDims={'HW':thisCnnHW, 
                                                       'MapC': channelListCNN[jj],
                                                       'VitC': channelListVit[jj],
                                                       'VitDim': thisVitDim,
                                                       'PatchDim': thisPatchDim}, 
                                        outDims={'HW':thisCnnHW, 
                                                 'MapC': channelListCNN[jj+1],
                                                 'VitC': channelListVit[jj+1],
                                                 'VitDim': thisVitDim}, 
                                        config={'option': thisArchitecture,
                                                'downsample': False}).cuda()
                        thisResList.append(thisResult)
                    self.fusingOperations.append(nn.Sequential(*thisResList))
                else:
                    self.fusingOperations.append(nn.Identity())
        else:
            for ii in range(len(self.encodingFeatureShape)):
                if ii==0:
                    continue
                self.fusingOperations.append(nn.Identity())
 

    # Method to fuse style features using the selected fusing method
    def FusingStyleFeatures(self, styleFeatures):
        fusingMethod = StyleFusingDict[self.styleFusingMethod]
        fusedFeatures=[]
        for ii in range(len(styleFeatures)):            
            for jj in range(len(styleFeatures[ii])):
                if jj ==0:
                    cnnAdded = torch.unsqueeze(styleFeatures[ii][jj].cnn,0)
                    vitAdded = torch.unsqueeze(styleFeatures[ii][jj].vit,0)
                else:
                    cnnAdded=torch.concat((cnnAdded,torch.unsqueeze(styleFeatures[ii][jj].cnn,0)),dim=0)
                    vitAdded=torch.concat((vitAdded,torch.unsqueeze(styleFeatures[ii][jj].vit,0)),dim=0)
            cnnFused = fusingMethod(cnnAdded,dim=0)[0]
            vitFused = fusingMethod(vitAdded,dim=0)[0]
            thisFused = BlockFeature(cnn=cnnFused, vit=vitFused)
            fusedFeatures.append(thisFused)
        return fusedFeatures
        
    # Method to fuse content and style features
    def FusingContentAndStyle(self, styles, contents):
        fusedList=[]
        for ii in range(len(styles)):
            # if ii==0:
            #     continue # Skip the STEM feature
            
            thisFusedCNN = torch.concat((styles[ii].cnn, contents[ii].cnn),dim=1)
            thisFusedVIT = torch.concat((styles[ii].vit, contents[ii].vit),dim=-1)
            thisFused = BlockFeature(cnn=thisFusedCNN, vit=thisFusedVIT)
            fusedList.append(thisFused)
        return fusedList
            
    
    # Forward pass for mixing content and style features
    def forward(self, styleFeatures, contentFeatures):
        # Skip the STEM features during the fusion process
        styleFeatures=styleFeatures[1:]     # Skip the STEM features
        contentFeatures=contentFeatures[1:] # Skip the STEM features
        
        # Fuse style features and then fuse content and style together
        fusedStyleFeatureList = self.FusingStyleFeatures(styleFeatures=styleFeatures)
        fusedContentStyle = self.FusingContentAndStyle(styles=fusedStyleFeatureList, contents=contentFeatures)
        
        fusedFinalList=[]
        for ii in range(len(fusedContentStyle)):
            thisFused = self.fusingOperations[ii](fusedContentStyle[ii])
            fusedFinalList.append(thisFused)
        
        return fusedFinalList

