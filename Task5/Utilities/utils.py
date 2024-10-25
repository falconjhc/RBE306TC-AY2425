# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
import copy as cp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import pylab
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import os
import random
import numpy as np
import torch
from PIL import Image
import logging
from datetime import datetime



LABEL_LENGTH=6
class Logging(logging.StreamHandler):
    """
    Custom StreamHandler that avoids adding a newline to logging messages.
    """
    def emit(self, record):
        try:
            # Ensure the record's message is a string
            msg = self.format(record)
            if not isinstance(msg, str):
                msg = str(msg)

            stream = self.stream
            if not getattr(self, 'terminator', '\n'):  # If terminator is set to empty
                stream.write(msg)
            else:
                stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def PrintInfoLog(handler, message, end='\n', dispTime=True):
    handler.terminator = end  # Set the terminator
    
    # Get the current date and time with full details
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Log message with full timestamp if it's a new line, else continue the same line
    if dispTime and end == '\n':
        logging.info(f"[{current_time}] {str(message)}")
    else:
        logging.info(str(message))



    
def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)

def SplitName(inputStr):
    result = []
    temp = ""
    for i in inputStr:
        if i.isupper() :
            if temp != "":
                result.append(temp)
            temp = i
        else:
            temp += i

    if temp != "":
        result.append(temp)
    return result

def write_to_file(path,write_list):
    file_handle = open(path,'w')
    for write_info in write_list:
        file_handle.write(str(write_info))
        file_handle.write('\n')
    file_handle.close()
    print("Write to File: %s" % path)

def read_from_file(path):
    # get label0 for the targeted content input txt
    output_list = list()
    with open(path) as f:
        for line in f:
            this_label = line[:-1]
            if len(this_label)<LABEL_LENGTH and not this_label == '-1':
                    for jj in range(LABEL_LENGTH-len(this_label)):
                        this_label = '0'+ this_label
            # line = u"%s" % line
            output_list.append(this_label)


    return output_list

def read_file_to_dict(file_path):
    line_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):
            line = int(line)
            line = str(line)
            # 移除每行末尾的换行符并存储到字典
            line_dict[line] = line_number
    return line_dict


def image_show(img):
    img_out = cp.deepcopy(img)
    img_out = np.squeeze(img_out)
    img_shapes=img_out.shape
    if len(img_shapes)==2:
        curt_channel_img = img_out
        min_v = np.min(curt_channel_img)
        curt_channel_img = curt_channel_img - min_v
        max_v = np.max(curt_channel_img)
        curt_channel_img = curt_channel_img/ np.float32(max_v)
        img_out = curt_channel_img*255
    elif img_shapes[2] == 3:
        channel_num = img_shapes[2]
        for ii in range(channel_num):
            curt_channel_img = img[:,:,ii]
            min_v = np.min(curt_channel_img)
            curt_channel_img = curt_channel_img - min_v
            max_v = np.max(curt_channel_img)
            curt_channel_img = curt_channel_img / np.float32(max_v)
            img_out[:,:,ii] = curt_channel_img*255
    else:
        print("Channel Number is INCORRECT:%d" % img_shapes[2])
    plt.imshow(np.float32(img_out)/255)
    pylab.show()

def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file





def correct_ckpt_path(real_dir,maybe_path):
    maybe_path_dir = str(os.path.split(os.path.realpath(maybe_path))[0])
    if not maybe_path_dir == real_dir:
        return os.path.join(real_dir,str(os.path.split(os.path.realpath(maybe_path))[1]))
    else:
        return maybe_path



def softmax(x):
    x -= np.max(x, axis= 1, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x


def create_if_not(path):
    #create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)
    

    

def cv2torch(file_path,transform):
    return transform(Image.open(file_path).convert('L'))

def string2tensor(string):
    return torch.tensor(int(string))

def set_random(seed_id=1234):
    #set random seed for reproduce
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_file_to_dict(file_path):
    line_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):
            line = int(line)
            line = str(line)
            # 移除每行末尾的换行符并存储到字典
            line_dict[line] = line_number
    return line_dict

def unormalize(tensor):
    # 反归一化操作
    tensor = tensor * 0.5 + 0.5  # 将 [-1, 1] 范围的值映射回 [0, 1]
    # 转换为 numpy 数组并且确保类型为 uint8
    output = tensor.int()
    return output

