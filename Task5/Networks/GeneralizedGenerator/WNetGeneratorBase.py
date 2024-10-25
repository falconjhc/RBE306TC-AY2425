# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function


import sys
sys.path.append('../')
sys.path.append('../../')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import torch
import torch.nn as nn

torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()
from Utilities.utils import SplitName



from Networks.GeneralizedGenerator.GeneralizeEncoder import GeneralizedEncoder as Encoder
from Networks.GeneralizedGenerator.GeneralizedMixer import WNetMixer as Mixer
from Networks.GeneralizedGenerator.GeneralizedDecoder import GeneralizedDecoder as Decoder
from Utilities.utils import set_random
from prettytable import PrettyTable
from Utilities.utils import PrintInfoLog


eps = 1e-9
generator_dim = 64
print_separater="#########################################################"

# WNetGeneratorBase is the main network class that constructs the generator 
class WNetGeneratorBase(nn.Module):
    def __init__(self, config, sessionLog):
        super(WNetGeneratorBase, self).__init__()    
        self.sessionLog = sessionLog
        
        # Configuration settings
        self.config=config
        self.is_train=True
        set_random()
        
        # Parse the encoder and decoder architectures
        self.encodingArchitectureList=SplitName(self.config.generator.encoder)[1:]
        self.decodingArchitectureList=SplitName(self.config.generator.decoder)[1:]
        self.encodingBlockNum=len(self.encodingArchitectureList)+1
        
        # Initialize content and style encoders
        self.contentEncoder=Encoder(config=config,
                                     loadedCategoryLength=len(config.datasetConfig.loadedLabel0Vec),
                                     mark='Content')
        self.contentEncoder.train()
        self.contentEncoder.cuda()
        
        self.styleEncoder=Encoder(config=config,
                                  loadedCategoryLength=len(config.datasetConfig.loadedLabel1Vec),
                                  mark='Style')
        self.styleEncoder.train()
        self.styleEncoder.cuda()

        # Test the encoders to verify architecture
        contentFeatures,styleFeatureList,contentCategory, styleCategoryList, encodingFeatureShape = self.TestEncoders()
        
        # Initialize the mixer
        self.mixer = Mixer(encodingFeatureShape=encodingFeatureShape, config=self.config)
        self.mixer.train()
        self.mixer.cuda()
        
        # Test the mixer
        fusedFinal, mixerFeatureShape=self.TestMixer(styleFeatures=styleFeatureList, contentFeatures=contentFeatures)
        
        # Initialize and test the decoder
        self.decoder = Decoder(config=config, mixerFeatureShape=mixerFeatureShape)
        self.decoder.train()
        self.decoder.cuda()
        
        _ = self.TestDecoder(fusedFinal)
        PrintInfoLog(self.sessionLog, print_separater)
        PrintInfoLog(self.sessionLog, "Architecture Construction Completed")
        PrintInfoLog(self.sessionLog, print_separater)
        



        
    # Testing the encoders to verify each layer's output
    def TestEncoders(self):
        testContent = torch.randn(self.config.trainParams.batchSize, self.config.datasetConfig.inputContentNum, 64, 64).to('cuda')  
        testStyle = torch.randn(self.config.trainParams.batchSize, self.config.datasetConfig.inputStyleNum, 64, 64).to('cuda')  
        
        # Process content encoders
        contentCategory, contentFeatures = self.contentEncoder(testContent)
        PrintInfoLog(self.sessionLog, print_separater)
        PrintInfoLog(self.sessionLog, "Content Encoder Architecture", dispTime=False)
        table = PrettyTable(['Layer', 'CNN','VIT'])
        table.add_row(['0-STEM-Bottleneck']+ contentFeatures[0].ProcessOutputToList() )
        table.add_row(['1-'+self.encodingArchitectureList[0]]+ contentFeatures[1].ProcessOutputToList() )
        table.add_row(['2-'+self.encodingArchitectureList[1]]+ contentFeatures[2].ProcessOutputToList() )
        table.add_row(['3-'+self.encodingArchitectureList[2]]+ contentFeatures[3].ProcessOutputToList() )
        table.add_row(['4-'+self.encodingArchitectureList[3]]+ contentFeatures[4].ProcessOutputToList() )
        PrintInfoLog(self.sessionLog, table, dispTime=False)     
        
        encodingFeatureShape = [contentFeatures[0].ProcessOutputToList(),
                                contentFeatures[1].ProcessOutputToList(),
                                contentFeatures[2].ProcessOutputToList(),
                                contentFeatures[3].ProcessOutputToList(),
                                contentFeatures[4].ProcessOutputToList()]
        
        # Process style encoders
        styleCategoryList=[]
        styleFeatureList=[]
        for ii in range(self.encodingBlockNum):
            styleFeatureList.append([])
        for ii in range(self.config.datasetConfig.inputStyleNum):
            thisStyleInput = torch.unsqueeze(testStyle[:,ii,:,:],1)
            thisStyleCategory, thisStyleFeatures = self.styleEncoder(thisStyleInput)
            styleCategoryList.append(thisStyleCategory)
            
            
            for jj in range(len(thisStyleFeatures)):
                styleFeatureList[jj].append(thisStyleFeatures[jj])
        PrintInfoLog(self.sessionLog, print_separater)
        PrintInfoLog(self.sessionLog, "Style Encoder Architecture", dispTime=False)
        table = PrettyTable(['Layer', 'CNN','VIT'])
        table.add_row(['0-STEM-Bottleneck']+ styleFeatureList[0][0].ProcessOutputToList() )
        table.add_row(['1-'+self.encodingArchitectureList[0]]+ styleFeatureList[1][0].ProcessOutputToList())
        table.add_row(['2-'+self.encodingArchitectureList[1]]+ styleFeatureList[2][0].ProcessOutputToList())
        table.add_row(['3-'+self.encodingArchitectureList[2]]+ styleFeatureList[3][0].ProcessOutputToList())
        table.add_row(['4-'+self.encodingArchitectureList[3]]+ styleFeatureList[4][0].ProcessOutputToList())
        PrintInfoLog(self.sessionLog, table, dispTime=False)     
        PrintInfoLog(self.sessionLog, print_separater, dispTime=False)
        return contentFeatures,styleFeatureList,contentCategory, styleCategoryList, encodingFeatureShape
        
        
    # Test the Mixer architecture by fusing content and style features
    def TestMixer(self, styleFeatures, contentFeatures):
        fusedFinal=self.mixer(styleFeatures=styleFeatures, contentFeatures=contentFeatures)
        PrintInfoLog(self.sessionLog, print_separater)
        PrintInfoLog(self.sessionLog, "Mixer Architecture", dispTime=False)
        table = PrettyTable(['Layer', 'CNN','VIT'])
        table.add_row(['1-'+self.encodingArchitectureList[0]]+ fusedFinal[0].ProcessOutputToList() )
        table.add_row(['2-'+self.encodingArchitectureList[1]]+ fusedFinal[1].ProcessOutputToList() )
        table.add_row(['3-'+self.encodingArchitectureList[2]]+ fusedFinal[2].ProcessOutputToList() )
        table.add_row(['4-'+self.encodingArchitectureList[3]]+ fusedFinal[3].ProcessOutputToList() )
        PrintInfoLog(self.sessionLog, table, dispTime=False)     
        mixerFeatureShape = [fusedFinal[0].ProcessOutputToList(),
                                fusedFinal[1].ProcessOutputToList(),
                                fusedFinal[2].ProcessOutputToList(),
                                fusedFinal[3].ProcessOutputToList()]
        return fusedFinal, mixerFeatureShape
    
    # Test the Decoder and verify its output shapes
    def TestDecoder(self, encoded):
        decoderList=self.decoder(encoded)
        generated = decoderList[0]
        
        PrintInfoLog(self.sessionLog, print_separater)
        PrintInfoLog(self.sessionLog, "Decoder Architecture", dispTime=False)
        table = PrettyTable(['CNN','VIT','Layer'])
        table.add_row([str(generated.shape)[11:-1]]+['NA']+ ['5-Generated-BasicBlock'])
        table.add_row(decoderList[1].ProcessOutputToList() +['4-'+self.decodingArchitectureList[-1]])
        table.add_row(decoderList[2].ProcessOutputToList() +['3-'+self.decodingArchitectureList[-2]])
        table.add_row(decoderList[3].ProcessOutputToList() +['2-'+self.decodingArchitectureList[-2]])
        table.add_row(decoderList[4].ProcessOutputToList() +['1-'+self.decodingArchitectureList[-3]])
        PrintInfoLog(self.sessionLog, table, dispTime=False)     
        return generated
    
    # Forward pass through the generator
    def forward(self,content_inputs,style_inputs,GT, is_train=True):
        contentCategory_onReal, contentFeatures_onReal = self.contentEncoder(content_inputs)        
        styleCategoryFull_onReal=[]
        styleFeatureList_onReal=[]
        for ii in range(self.encodingBlockNum):
            styleFeatureList_onReal.append([])
        
        # Encode each style image separately
        style_inputs = style_inputs.reshape((self.config.trainParams.batchSize, self.config.datasetConfig.inputStyleNum, 
                                             self.config.datasetConfig.imgWidth,self.config.datasetConfig.imgWidth))
        for ii in range(self.config.datasetConfig.inputStyleNum):
            this_style_category_onReal, this_style_outputs_onReal = self.styleEncoder(torch.unsqueeze(style_inputs[:,ii,:,:], dim=1))
            if ii ==0:
                styleCategoryFull_onReal = this_style_category_onReal
            else:
                styleCategoryFull_onReal = torch.concat((styleCategoryFull_onReal,this_style_category_onReal), dim=0)
            for jj in range(len(this_style_outputs_onReal)):
                styleFeatureList_onReal[jj].append(this_style_outputs_onReal[jj])
        
        
        reshaped_style_list_onReal=[]
        for ii in range(len(styleFeatureList_onReal)):
            for jj in range(len(styleFeatureList_onReal[ii])):
                if jj ==0:
                    thisFeature = torch.unsqueeze(styleFeatureList_onReal[ii][jj].cnn,1)
                else:
                    thisFeature = torch.concat((thisFeature,torch.unsqueeze(styleFeatureList_onReal[ii][jj].cnn,1)),1)
            reshaped_style_list_onReal.append(thisFeature)
        enc_content_list_onReal= [ii.cnn for ii in contentFeatures_onReal]

        # Mixing content and style features
        mix_output = self.mixer(styleFeatures=styleFeatureList_onReal, contentFeatures=contentFeatures_onReal)
        decode_output_list=self.decoder(mix_output)
        for ii in range(len(decode_output_list)):
            if ii ==0:
                continue
            else:
               decode_output_list[ii]=decode_output_list[ii].cnn 
        decode_output_list.reverse()
        generated = decode_output_list[-1]
        
        
        # Encode ground truth and generated outputs
        GT_content_category,GT_content_outputs = self.contentEncoder(GT.repeat(1, GT.shape[1]*self.config.datasetConfig.inputContentNum, 1, 1))
        GT_style_category,GT_style_outputs = self.styleEncoder(GT)
        contentCategoryOnGenerated, contentFeaturesOnGenerated = self.contentEncoder(generated.repeat((1,self.config.datasetConfig.inputContentNum,1,1)))
        enc_content_onGenerated_list= [ii.cnn for ii in contentFeaturesOnGenerated]
        styleCategoryOnGenerated, styleFeaturesOnGenerated = self.styleEncoder(generated)
        enc_style_onGenerated_list= [ii.cnn for ii in styleFeaturesOnGenerated]
        contentFeaturesOnGenerated=enc_content_onGenerated_list
        styleFeaturesOnGenerated = enc_style_onGenerated_list
        
        
        
        # Max-pooling across categories
        max_content_category_onReal = torch.max(self.TensorReshape(contentCategory_onReal, is_train),dim=1)[0] 
        max_style_category_onReal = torch.max(self.TensorReshape(styleCategoryFull_onReal, is_train),dim=1)[0]
        max_lossCategoryContentFakeerated = torch.max(self.TensorReshape(contentCategoryOnGenerated, is_train),dim=1)[0]
        max_lossCategoryStyleFakeerated = torch.max(self.TensorReshape(styleCategoryOnGenerated, is_train),dim=1)[0]
        
        encodedContentFeatures={}
        encodedStyleFeatures={}
        encodedContentCategory={}
        encodedStyleCategory={}
        
        encodedContentFeatures.update({'real': enc_content_list_onReal[-1]})
        encodedContentFeatures.update({'fake': contentFeaturesOnGenerated[-1]})
        encodedContentFeatures.update({'groundtruth': GT_content_outputs[-1].cnn})
        
        encodedStyleFeatures.update({'real': reshaped_style_list_onReal[-1]})
        encodedStyleFeatures.update({'fake': styleFeaturesOnGenerated[-1]})
        encodedStyleFeatures.update({'groundtruth': GT_style_outputs[-1].cnn})
        
        encodedContentCategory.update({'real': max_content_category_onReal})
        encodedContentCategory.update({'fake': max_lossCategoryContentFakeerated})
        encodedContentCategory.update({'groundtruth': GT_content_category})
        
        encodedStyleCategory.update({'real': max_style_category_onReal})
        encodedStyleCategory.update({'fake': max_lossCategoryStyleFakeerated})
        encodedStyleCategory.update({'groundtruth': GT_style_category})
        
        return encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated

    
    # Method to reshape tensors based on training mode and batch size
    def TensorReshape(self,input_tensor,is_train):
        if len(input_tensor.shape) == 4:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3])
        elif len(input_tensor.shape) == 3:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1],input_tensor.shape[2])
        elif len(input_tensor.shape) == 2:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1])