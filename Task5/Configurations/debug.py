
import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'seed':1,
        'debugMode':1,
        'expID':'Debug',# experiment name prefix
        'expDir': '/data-shared/server02/data1/haochuan/Character/DebugRecords202410/',
        
        # user interface
        'printInfoSeconds':3,

       

        'YamlPackage': '../YamlLists/Debug/',
        
       
        
        'FullLabel0Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/Debug-Label0.txt',
        'FullLabel1Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/Debug-Label1.txt',

        
        # training configurations
       'augmentation':'HardAumentation',
        
        
        'inputStyleNum':9, 
        'inputContentNum':5,

       
        'discriminator':'NA',


        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimizer':'rms',
        'gradNorm': 1,
        'initTrainEpochs':0,

        # feature extractor parametrers
        'TrueFakeExtractorPath': [],
        'ContentExtractorPath':['/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG11Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG13Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG16Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG19Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet18/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet34/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet50/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet101/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet152/BestExtractor.pth'],
        
        'StyleExtractorPath':  ['/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG11Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG13Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG16Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG19Net/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet18/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet34/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet50/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet101/BestExtractor.pth',
                                           '/data-shared/NAS/RBE306TC/AY2425/PretrainedWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet152/BestExtractor.pth']
        
        # 'ContentExtractorPath':[],
        # 'StyleExtractorPath':  []


}


penalties = {
        'PenaltyGeneratorWeightRegularizer': 0.0001,
        'PenaltyDiscriminatorWeightRegularizer':0.0003,
        'PenaltyReconstructionL1':1,
        'PenaltyConstContent':0.2,
        'PenaltyConstStyle':0.2,
        'PenaltyAdversarial': 0,
        'PenaltyDiscriminatorCategory': 0,
        'GeneratorCategoricalPenalty': 1,
        'PenaltyDiscriminatorGradient': 0,
        # 'PenaltyContentFeatureExtractor': [0.5,0.1,0.1,0.1,0.1,0.1],
        # 'PenaltyStyleFeatureExtractor':[1,0.5,0.5,0.3,0.3],
        'PenaltyContentFeatureExtractor': [1,1,1,1,1,1,1,1,1],
        'PenaltyStyleFeatureExtractor':[1,1,1,1,1,1,1,1,1],
        
        'GradientPenalty_L1_Content': 0.5,
        'GradientPenalty_L1_Style': 0.5,
        
        # 'PenaltyContentFeatureExtractor': [],
        # 'PenaltyStyleFeatureExtractor':[],
}

