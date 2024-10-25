import sys
import torch
torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

sys.path.append('./')
from Networks.FeatureExtractor.FeatureExtractorBase import FeatureExtractorBase as FeatureExtractor
from Networks.PlainGenerators.PlainWNetBase import WNetGenerator
HighLevelFeaturePenaltyPctg=[0.1,0.15,0.2,0.25,0.3]
eps = 1e-9
MAX_gradNorm=1.0
MIN_gradNorm=1e-1
from Utilities.utils import PrintInfoLog

# from Utilities.utils import LogReloaded as Log

import copy


class Loss(nn.Module):
    def __init__(self, config, sessionLog, penalty):
        super(Loss, self).__init__()
        
        self.sessionLog = sessionLog

        # penalities
        self.PenaltyReconstructionL1 = penalty['PenaltyReconstructionL1']
        self.PenaltyConstContent = penalty['PenaltyConstContent']
        self.PenaltyConstStyle = penalty['PenaltyConstStyle']
        self.GeneratorCategoricalPenalty = penalty['GeneratorCategoricalPenalty']
        self.PenaltyContentFeatureExtractor = penalty['PenaltyContentFeatureExtractor']
        self.PenaltyStyleFeatureExtractor = penalty['PenaltyStyleFeatureExtractor']

       
        self.contentExtractorList=[]
        if 'extractorContent' in config:
            PrintInfoLog(self.sessionLog, "Content Encoders: ", end='')
            counter=0
            for contentExtractor in config['extractorContent']:
                thisContentExtractor = FeatureExtractor(outputNums=len(config.datasetConfig.loadedLabel0Vec), 
                                                        modelSelect=contentExtractor.name,
                                                        type='content').extractor
                thisContentExtractor.eval()
                thisContentExtractor.cuda()
                self.NameMappingLoading(thisContentExtractor, contentExtractor.path)
                self.contentExtractorList.append(copy.deepcopy(thisContentExtractor))
                if counter != len(config['extractorContent'])-1:
                    PrintInfoLog(self.sessionLog, ", ",end='')
                counter+=1
            PrintInfoLog(self.sessionLog, " Loaded.")
                    
        self.styleExtractorList=[]
        if 'extractorStyle' in config:
            PrintInfoLog(self.sessionLog, "Style Encoders: ", end='')
            counter=0
            for styleExtractor in config['extractorStyle']:
                thisStyleExtractor = FeatureExtractor(outputNums=len(config.datasetConfig.loadedLabel1Vec), 
                                                        modelSelect=styleExtractor.name,
                                                        type='style').extractor
                thisStyleExtractor.eval()
                thisStyleExtractor.cuda()
                self.NameMappingLoading(thisStyleExtractor, styleExtractor.path)
                self.styleExtractorList.append(copy.deepcopy(thisStyleExtractor))
                if counter != len(config['extractorStyle'])-1:
                    PrintInfoLog(self.sessionLog, ", ",end='')
                counter+=1
            PrintInfoLog(self.sessionLog, " Loaded.")
     

        
    def NameMappingLoading(self,extractor, path):
        loaded = torch.load(path)
        loadedItems=list(loaded.items())
        thisExtractorDict=extractor.state_dict()
        count=0
        for key,value in thisExtractorDict.items():
            layer_name,weights=loadedItems[count]      
            thisExtractorDict[key]=weights
            count+=1
        extractor.load_state_dict(thisExtractorDict)
        PrintInfoLog(self.sessionLog, path.split('/')[-2], end='')
        
        


    def GeneratorLoss(self,  encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory,\
                        generated, GT, content_onehot,style_onehot,lossInputs=None):
        # l1 loss
        # lossL1 = F.lossL1(generated, GT, reduction='mean')
        lossL1 = torch.mean(torch.abs(generated-GT))
        
        

        # const_content loss
        GT_content_enc = encodedContentFeatures['groundtruth']
        lossConstContentReal = F.mse_loss(encodedContentFeatures['real'],GT_content_enc)
        lossConstContentFake = F.mse_loss(encodedContentFeatures['fake'],GT_content_enc)
        
        
        
        # const_style loss
        GT_style_enc = encodedStyleFeatures['groundtruth']
        reshaped_styles = encodedStyleFeatures['real'] # batchsize * input style num * channels * width * height
        reshaped_styles = reshaped_styles.permute(1,0,2,3,4)#  input style num *batchsize * channels * width * height
        lossConstStyleReal = [F.mse_loss(enc_style, GT_style_enc) for enc_style in reshaped_styles] # enc_style: batchsize * channels * width * height
        lossConstStyleReal = torch.mean(torch.stack(lossConstStyleReal))
        lossConstStyleFake = F.mse_loss(encodedStyleFeatures['fake'],GT_style_enc)            

        # category loss
        lossCategoryContentReal,lossCategoryContentFake = 0,0
        GT_content_category = encodedContentCategory['groundtruth']
        for fake_logits,GT_logits,onehot in zip(encodedContentCategory['real'],GT_content_category,content_onehot):
            GT_logits = torch.nn.functional.softmax(GT_logits, dim=0)
            lossCategoryContentReal += F.cross_entropy(GT_logits, onehot)
            fake_logits = torch.nn.functional.softmax(fake_logits, dim=0)            
            lossCategoryContentFake += F.cross_entropy(fake_logits, onehot)
        lossCategoryContentReal /= len(GT_content_category)
        lossCategoryContentFake /= len(GT_content_category)
        
        lossCategoryStyleReal,lossCategoryStyleFake = 0,0
        GT_style_category = encodedStyleCategory['groundtruth']
        for fake_logits,GT_logits,onehot in zip(encodedStyleCategory['real'],GT_style_category,style_onehot):
            GT_logits = torch.nn.functional.softmax(GT_logits, dim=0)
            lossCategoryStyleReal += F.cross_entropy(GT_logits, onehot)
            fake_logits = torch.nn.functional.softmax(fake_logits, dim=0)            
            lossCategoryStyleFake += F.cross_entropy(fake_logits, onehot)
        lossCategoryStyleReal /= len(GT_style_category)
        lossCategoryStyleFake /= len(GT_style_category)
        
        
        losses = [lossL1,lossConstContentReal,lossConstStyleReal,lossConstContentFake,lossConstStyleFake,\
            lossCategoryContentReal,lossCategoryContentFake,lossCategoryStyleReal,lossCategoryStyleFake]
        
        
        return losses

    def FeatureExtractorLoss(self,GT,imgFake):
        # content_extractor
        contentSumMSE=0.0
        contentMSEList=[]
        # travel for different feature extractors
        for idx1, thisContentExtractor in enumerate(self.contentExtractorList):
            thisContentMSE=0
            with torch.no_grad():
                _,GT_content_features = thisContentExtractor(GT)
                _,fake_content_features = thisContentExtractor(imgFake)
            if not len(HighLevelFeaturePenaltyPctg) == \
                    len(GT_content_features) == \
                    len(fake_content_features):
                PrintInfoLog(self.sessionLog, 'content length not paired')
                return
            
            # travel for different evaluating layers
            for idx2,(GT_content_feature,fake_content_feature) in enumerate(zip(GT_content_features,fake_content_features)):
                thisContentMSE += F.mse_loss(GT_content_feature,fake_content_feature)*HighLevelFeaturePenaltyPctg[idx2]
            thisContentMSE /= sum(HighLevelFeaturePenaltyPctg)
            contentMSEList.append(thisContentMSE)
            contentSumMSE+=thisContentMSE*self.PenaltyContentFeatureExtractor[idx1]
        contentSumMSE = contentSumMSE / (sum(self.PenaltyContentFeatureExtractor)+eps)


        # style_extractor
        styleSumMSE=0.0
        styleMSEList=[]
        for idx1, thsiStyleExtractor in enumerate(self.styleExtractorList):
            thisStyleMSE = 0
            with torch.no_grad():
                _,GT_style_features = thsiStyleExtractor(GT)
                _,fake_style_features = thsiStyleExtractor(imgFake)
            if not len(HighLevelFeaturePenaltyPctg) == \
                    len(GT_style_features) == \
                    len(fake_style_features):
                PrintInfoLog(self.sessionLog, 'style length not paired')
                return
            for idx2,(GT_style_feature,fake_style_feature) in enumerate(zip(GT_style_features,fake_style_features)):
                thisStyleMSE += F.mse_loss(GT_style_feature,fake_style_feature)*HighLevelFeaturePenaltyPctg[idx2]
            thisStyleMSE /= sum(HighLevelFeaturePenaltyPctg)
            styleMSEList.append(thisStyleMSE)
            styleSumMSE+=thisStyleMSE*self.PenaltyStyleFeatureExtractor[idx1]
        styleSumMSE = styleSumMSE / (sum(self.PenaltyStyleFeatureExtractor)+eps)

        return contentSumMSE,styleSumMSE,contentMSEList,styleMSEList



    def forward(self, encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory,\
        generated, GT, content_onehot,style_onehot,lossInputs=None):
        # generator_const_loss
        losses = \
            self.GeneratorLoss(encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory,\
                        generated, GT, content_onehot,style_onehot, lossInputs=lossInputs)
        lossL1,\
            lossConstContentReal,lossConstStyleReal,\
                lossConstContentFake,lossConstStyleFake, \
                    lossCategoryContentReal,lossCategoryContentFake,\
                        lossCategoryStyleReal,lossCategoryStyleFake = losses
                        
        
            
            
        
        # generator_category_loss
        deepPerceptualContentSum,deepPerceptualStyleSum,contentMSEList,styleMSEList =\
            self.FeatureExtractorLoss(GT=GT,imgFake=generated)

    
        sumLossG = lossL1 * self.PenaltyReconstructionL1 + \
                    (lossConstContentReal+lossConstContentFake) * self.PenaltyConstContent + \
                    (lossConstStyleReal+lossConstStyleFake) * self.PenaltyConstStyle 
                
                    
        lossDict = {'lossL1':lossL1,
                    'lossConstContentReal':lossConstContentReal,
                    'lossConstStyleReal':lossConstStyleReal,
                    'lossConstContentFake':lossConstContentFake,
                    'lossConstStyleFake':lossConstStyleFake,
                    'lossCategoryContentReal':lossCategoryContentReal,
                    'lossCategoryContentFake':lossCategoryContentFake,
                    'lossCategoryStyleReal':lossCategoryStyleReal,
                    'lossCategoryStyleFake':lossCategoryStyleFake,
                    'deepPerceptualContent':deepPerceptualContentSum,
                    'deepPerceptualStyle':deepPerceptualStyleSum,
                    'deepPerceptualContentList': contentMSEList,
                    'deepPerceptualStyleList': styleMSEList}


        return sumLossG,lossDict

