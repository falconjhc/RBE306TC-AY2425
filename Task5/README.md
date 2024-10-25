# Implementation of the GeneralizedWNet with Pytorch

First of all, Pytorch is not that easy to be used: particularly in the scenario when you have been familiar with static graphs in the Tensorflow 1.x

This repository contains the implementation of the Generalised WNet, which is a transplanted source code repository from [GeneralizedWNet-Tensorflow1.x](https://github.com/falconjhc/GeneralizedWNet-Tensorflow1.x).

## Installation
To set up the environment, you can create a Conda environment and install the required packages using a `requirements.txt` file.

1. **Create a Conda Environment:**
```bash
conda create -n pytorch-character python=3.8
```
   
2. **Activate the Conda Environment:**

```bash
conda activate pytorch-character
```

3. **Install Required Packages:**
Ensure you have a requirements.txt file in the root of your project directory. If you don’t have one, you can create it using pip freeze. Then, run the following command:
```bash
pip install -r requirements.txt
```


## Training Instructions

You can train the model using the following commands based on your requirements:

### 1. Train a Plain WNet:

```bash
cd Scripts
python PipelineScripts.py --wnet plain --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 11 --resumeTrain 1 --config PF64-PF50 --device 0
```

### 2. Train a Generalized WNet with Basic ResBlocks:

```bash
python PipelineScripts.py --wnet general --encoder EncoderCvCvCvCv --decoder DecoderCvCvCvCv --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 55 --resumeTrain 0 --config PF64-PF50 --device 0
```


### 3. Train a Generalized WNet with Bottleneck ResBlocks:
```bash
python PipelineScripts.py --wnet general --encoder EncoderCbnCbnCbnCbn --decoder DecoderCbnCbnCbnCbn --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 55 --resumeTrain 0 --config PF64-PF50 --device 0
```


### 4. Train a Generalized WNet with Vision Transformers:
```bash
python PipelineScripts.py --wnet general --encoder EncoderCbnCbnCbnVit@2@24 --decoder DecoderVit@2@24CbnCbnCbn --mixer MixerMaxRes7@3 --batchSize 64 --initLr 0.001 --epochs 55 --resumeTrain 0 --config PF64-PF50 --device 0
```

###  5. Congurations:
To find configurations, please navigate to the directory:
```bash
[where you place your code]/Configurations/[some configuration file].py 
```

## Argument Instructions

###  Debugging
To enable debugging, add the --debug 1 flag to your command. Remove it when running the code normally.

### Skip Testing 
To skip that boring testings, add the --skipTest 1 flag to your command, which make your debugging process much quicker. Remove it when you change your mind. 





## Note
This repository is still under construction, and updates are expected from time to time.
Enjoy!

```bash
Feel free to modify any part of it to better suit your needs!
```