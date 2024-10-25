import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'seed':1,
        'debugMode': 0,
        'expID':'20240814-ContentPF64-StyleHW50',# experiment name prefix
        'expDir': '/data-shared/server02/data1/haochuan/Character/TrainRecords-202408/',
        


        # devices
        'generatorDevice':'/device:GPU:0',
        'discrminatorDevice':'/device:GPU:0',
        'featureExtractorDevice':'/device:GPU:0',

        
        # data 
        'content_data_dir':# standard data location
        [os.path.join(dataPathRoot, 'CASIA_Dataset/PrintedData_64Fonts/Simplified/GB2312_L1/')],

        'style_train_data_dir': # training data location
        [os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB1.1/'),
         os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB2.1/')],

        'style_validation_data_dir':# validation data location
        [os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB1.1/'),
         os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB2.1/')],

        'file_list_txt_content': # file list of the standard data
        ['../FileList/PrintedData/Char_0_3754_64PrintedFonts_GB2312L1_Simplified.txt'],

        'file_list_txt_style_train': # file list of the training data
        ['../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1101_1150_Isolated_Train.txt',
         '../TrainTestFileList/HandWritingData/Char_0_3754_Writer_1101_1150_Cursive_Train.txt'],

        'file_list_txt_style_validation': # file list of the validation data
        ['../TrainTestFileList/HandWritingData/ForTrain_Char_0_3754_Writer_1151_1200_Isolated_Test.txt',
         '../TrainTestFileList/HandWritingData/ForTrain_Char_0_3754_Writer_1151_1200_Cursive_Test.txt'],

        'FullLabel0Vec': 'CASIA_Dataset/LabelVecs/HW300-Label0.txt',
        'FullLabel1Vec': 'CASIA_Dataset/LabelVecs/HW300-Label1.txt',
        
        
        # training configurations
        'trainAugmentContentTranslation':1,
        'trainAugmentContentRotation':1,
        'trainAugmentContentFlip':1,
        'trainAugmentStyleTranslation':1,
        'trainAugmentStyleRotation':1,
        'trainAugmentStyleFlip':1,
        'trainSplitContentStyleAugmentation': 1,

        'testAugmentContentTranslation':0,
        'testAugmentContentRotation':0,
        'testAugmentContentFlip':0,
        'testAugmentStyleTranslation':0,
        'testAugmentStyleRotation':0,
        'testAugmentStyleFlip':0,
        'testSplitContentStyleAugmentation':0,
        
        'inputStyleNum':5, 

        # generator && discriminator
        'generator': 'TransWNet-AvgMaxConvResidualMixer-BasicBlockDecoder',
        'discriminator':'NA',

        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimizer':'adam',
        'initTrainEpochs':0,
        'final_learning_rate_pctg': 0.30,

        # feature extractor parametrers
        'TrueFakeExtractorPath': [],
        
        'ContentExtractorPath':
        [
        # '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_VGG16-FeatureExtractor_Content_PF64_vgg16net/variables/@/device:GPU:0',
        # '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_VGG19-FeatureExtractor_Content_PF64_vgg19net/variables/@/device:GPU:0',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet18-FeatureExtractor_Content_PF64_resnet18/variables/@/device:GPU:0',
        # '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet34-FeatureExtractor_Content_PF64_resnet34/variables/@/device:GPU:0',
        #'/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet50-FeatureExtractor_Content_PF64_resnet50/variables/@/device:GPU:1',
        #'/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet101-FeatureExtractor_Content_PF64_resnet101/variables/@/device:GPU:1'
        ],
        
        'StyleExtractorPath':
       [
        # '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/HW300/Models/checkpoint/Exp20240311_VGG16-FeatureExtractor_Style_HW300_vgg16net/variables/@/device:GPU:0',
        # '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/HW300/Models/checkpoint/Exp20240311_VGG19-FeatureExtractor_Style_HW300_vgg19net//variables/@/device:GPU:0',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/HW300/Models/checkpoint/Exp20240311_ResNet18-FeatureExtractor_Style_HW300_resnet18/variables/@/device:GPU:0',
        # '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/HW300/Models/checkpoint/Exp20240311_ResNet34-FeatureExtractor_Style_HW300_resnet34/variables/@/device:GPU:0',
        #'/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/HW300/Models/checkpoint/Exp20240308_ResNet50-FeatureExtractor_Content_PF80_resnet50/variables/@/device:GPU:0'
        
        ]

}


penalties = {
        'PenaltyGeneratorWeightRegularizer': 0.0001,
        'PenaltyDiscriminatorWeightRegularizer':0.0003,
        'PenaltyReconstructionL1':1,
        'PenaltyConstContent':0.2,
        'PenaltyConstStyle':0.2,
        'PenaltyAdversarial': 0,
        'PenaltyDiscriminatorCategory': 0,
        'GeneratorCategoricalPenalty': 0.,
        'PenaltyDiscriminatorGradient': 0,
        'Batch_StyleFeature_Discrimination_Penalty':0,
        # 'PenaltyContentFeatureExtractor': [0.5, 0.3, 0.3,0.2],
        # 'PenaltyStyleFeatureExtractor':[1, 0.5, 0.3,0.2]
        'PenaltyContentFeatureExtractor': [0],
        'PenaltyStyleFeatureExtractor':[0]
}