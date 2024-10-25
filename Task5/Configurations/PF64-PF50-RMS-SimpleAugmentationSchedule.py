import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'seed':1,
        'debugMode':1,
        'expID':'20241023-Torch-PF64PF50-RMS-NormGrad-SimpleAugmentationSchecule',# experiment name prefix
        'expDir': '/data-shared/server02/data1/haochuan/Character/Records20241022/',
        
        # user interface
        'printInfoSeconds':3,

        'YamlPackage': '../YamlLists/PF64-PF80/',
        
        'FullLabel0Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/PF64-Label0.txt',
        'FullLabel1Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label1.txt',

        
        # training configurations
        'augmentation':'SimpleAugmentationSchecule', 
        # Options: ‘NoAugmentation’, 'SimpleAugmentation', 'HardAumentation', 'SimpleAugmentationSchecule', 'HardAugmentationSchecule'
        
        'inputStyleNum':5, 
        'inputContentNum':64,

        'discriminator':'NA',


        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimizer':'rms',
        'gradNorm': 1,
        'initTrainEpochs':0,
        'final_learning_rate_pctg':0.01,

        # feature extractor parametrers
        'TrueFakeExtractorPath': [],
        'ContentExtractorPath':['/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG11Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG13Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG16Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG19Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet18/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet34/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet50/BestExtractor.pth',
                                        #    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet101/BestExtractor.pth',
                                        #    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet152/BestExtractor.pth'
                                           ],
        
        'StyleExtractorPath':  ['/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG11Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG13Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG16Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG19Net/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet18/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet34/BestExtractor.pth',
                                           '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet50/BestExtractor.pth',
                                        #    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet101/BestExtractor.pth',
                                        #    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet152/BestExtractor.pth'
                                           ]
        
        # 'ContentExtractorPath':[],
        # 'StyleExtractorPath':  []


}


penalties = {
        'PenaltyGeneratorWeightRegularizer': 0.0001,
        'PenaltyDiscriminatorWeightRegularizer':0.0003,
        'PenaltyReconstructionL1':3,
        'PenaltyConstContent':0.2,
        'PenaltyConstStyle':0.2,
        'PenaltyAdversarial': 0,
        'PenaltyDiscriminatorCategory': 0,
        'GeneratorCategoricalPenalty': 0.,
        'PenaltyDiscriminatorGradient': 0,
        'Batch_StyleFeature_Discrimination_Penalty':0,
        
 
        
        
        'PenaltyContentFeatureExtractor': [1,1,1,1,1,1,1,
                                                #      1,1
                                                     ],
        'PenaltyStyleFeatureExtractor':[1,1,1,1,1,1,1,
                                                #   1,1
                                                  ],
        
        # 'PenaltyContentFeatureExtractor': [],
        # 'PenaltyStyleFeatureExtractor':[],
}

