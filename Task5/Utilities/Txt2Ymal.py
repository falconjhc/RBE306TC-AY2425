import os
from tqdm import tqdm
import yaml


## Reading
dataPathRoot= '/data0/haochuan/'
contentDir = 'CASIA_Dataset/PrintedData_64Fonts/Simplified/GB2312_L1/'
styleDir = 'CASIA_Dataset/PrintedData/GB2312_L1/'
content_list = '/data1/haochuan/Codes/WNet-20240614-NewAttepmt01/FileList/PrintedData/Char_0_3754_64PrintedFonts_GB2312L1_Simplified.txt'
train_list= '/data1/haochuan/Codes/WNet-20240614-NewAttepmt01/TrainTestFileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1_Train.txt'
val_list= '/data1/haochuan/Codes/WNet-20240614-NewAttepmt01/TrainTestFileList/PrintedData/Char_0_3754_Font_50_79_GB2312L1_Test.txt'
label0 = '/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label0.txt'
label1 = '/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label1.txt'
## Writing
trainStyleRef = './YamlListsTest/PF64-PF80/TrainStyleReference.yaml'
trainGT = './YamlListsTest/PF64-PF80/TrainGroundTruth.yaml'
testStyleRef = './YamlListsTest/PF64-PF80/TestStyleReference.yaml'
testGT = './YamlListsTest/PF64-PF80/TestGroundTruth.yaml'
contentSavePath = './YamlListsTest/PF64-PF80/Content.yaml'






def readtxt(path):
    res = []
    f = open(path,'r')
    lines = f.readlines()
    for ll in lines:
        res.append(ll.replace('\n',''))
    f.close()
    return res

if __name__ == '__main__':
    label0_list = readtxt(label0)
    label1_list = readtxt(label1)
    dataset = readtxt(train_list)
    content_dir = {content:[] for content in label0_list}
    style_dir = {str(int(index)):[] for index in label1_list}
    iteration_dir = {x.split("@")[3]:[] for x in dataset}

    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        if content in label0_list:
            content_dir[content].append(name)
        if style in style_dir.keys():    
            style_dir[style].append(os.path.join(os.path.join(dataPathRoot, styleDir),name))
        iteration_dir[name] = [os.path.join(os.path.join(dataPathRoot, styleDir), splits[-1]), content, style]     
    print("Save trainset")
    with open(trainStyleRef, 'w') as file:
        yaml.dump(style_dir, file, allow_unicode=True)
    with open(trainGT, 'w') as file:
        yaml.dump(iteration_dir, file, allow_unicode=True)



    dataset = readtxt(val_list)
    content_dir = {content:[] for content in label0_list}
    style_dir = {str(int(index)):[] for index in label1_list}
    iteration_dir = {x.split("@")[3]:[] for x in dataset}
    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        if content in label0_list:
            content_dir[content].append(name)
        if style in style_dir.keys():    
            style_dir[style].append(os.path.join(os.path.join(dataPathRoot, styleDir),name))
        iteration_dir[name] = [os.path.join(os.path.join(dataPathRoot, styleDir), splits[-1]), content, style]     
    print("Save testset")
    with open(testStyleRef, 'w') as file:
        yaml.dump(style_dir, file, allow_unicode=True)
    with open(testGT, 'w') as file:
        yaml.dump(iteration_dir, file, allow_unicode=True)

    dataset = readtxt(content_list)
    content_dir = {content:[] for content in label0_list}
    style_dir = {str(int(index)):[] for index in label1_list}
    iteration_dir = {x.split("@")[3]:[] for x in dataset}

    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        if content in label0_list:
            content_dir[content].append(os.path.join(os.path.join(dataPathRoot, contentDir),name))
        # if style in style_dir.keys():    
        #     style_dir[style].append(name)
        # iteration_dir[name] = [content,style]   
    print("Save prototypeset")
    # 将content_dir保存为YAML文件
    with open(contentSavePath, 'w') as file:
        yaml.dump(content_dir, file, allow_unicode=True)
