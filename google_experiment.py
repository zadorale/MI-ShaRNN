from __future__ import print_function
import sys
import subprocess
import os
import torch
import numpy as np
try:
  from python_speech_features import fbank
  print("[0m[1;37;44m"+"python_speech_features already installed"+"[0m")
except:
  print("[0m[1;37;41m"+"installing python_speech_features"+"[0m")
  subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                         'python_speech_features'])
try:
  import wget
  print("[0m[1;37;44m"+"wget already installed"+"[0m")
except:
  print("[0m[1;37;41m"+"installing wget"+"[0m")
  subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                         'wget'])

import myUtils
myUtils.directory_create(myUtils.GoogleSpeech)
myUtils.download_file(myUtils.GoogleSpeech)
myUtils.extract_file(myUtils.GoogleSpeech)

if('file_test.npy' in os.listdir(myUtils.directory+"/GoogleSpeech/Extracted/")):
    print("[0m[1;37;41m"+'FEATURES ALREADY EXTRACTED'+"[0m")
else:
    exec(open('process_google.py').read())


print("[0m[1;37;42m"+'FEATURES EXTRACTED STARTING EXPERIMENT'+"[0m")

from microsoft_rnn import SRNN2
import ShaMIRNN_trainer4 as trainerP
import microsoft_utils as utils

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if(torch.cuda.is_available()):
    torch.cuda.empty_cache()
DATA_DIR = myUtils.directory+'/GoogleSpeech/Extracted/'

x_train_, y_train = np.load(DATA_DIR + 'x_train.npy'), np.load(DATA_DIR + 'y_train.npy')
x_val_, y_val = np.load(DATA_DIR + 'x_val.npy'), np.load(DATA_DIR + 'y_val.npy')
x_test_, y_test = np.load(DATA_DIR + 'x_test.npy'), np.load(DATA_DIR + 'y_test.npy')
# Mean-var normalize
mean = np.mean(np.reshape(x_train_, [-1, x_train_.shape[-1]]), axis=0)
std = np.std(np.reshape(x_train_, [-1, x_train_.shape[-1]]), axis=0)
std[std[:] < 0.000001] = 1
x_train_ = (x_train_ - mean) / std
x_val_ = (x_val_ - mean) / std
x_test_ = (x_test_ - mean) / std

x_train = np.swapaxes(x_train_, 0, 1)
x_val = np.swapaxes(x_val_, 0, 1)
x_test = np.swapaxes(x_test_, 0, 1)

y_train = np.concatenate((np.zeros((y_train.shape[0],1)),y_train),axis =1)
y_val = np.concatenate((np.zeros((y_val.shape[0],1)),y_val),axis =1)
y_test = np.concatenate((np.zeros((y_test.shape[0],1)),y_test),axis =1)

print("Train shape", x_train.shape, y_train.shape)
print("Val shape", x_val.shape, y_val.shape)
print("Test shape", x_test.shape, y_test.shape)

numTimeSteps = x_train.shape[0]
numInput = x_train.shape[-1]
numClasses = y_train.shape[1]

# Network Parameters
brickSize = 8
hiddenDim0 = 64
hiddenDim1 = 32
cellType = 'LSTM'
learningRate = 0.001
batchSize = 128
epochs = 3

numK = 1
lenK = numK
numTrimmed = 49
numSkip = 10

numIter = 5
numRounds = 5

resetAtrounds = True
easymode = True

params = (numInput, numClasses, hiddenDim0, hiddenDim1, cellType)

srnn2 = SRNN2(numInput, numClasses, hiddenDim0, hiddenDim1, cellType).to(device) 
trainer = trainerP.ShaMIRNNTrainer(srnn2, learningRate, params, lossType='xentropy', device=device)


trainer.train(brickSize, batchSize, epochs, x_train, x_val, y_train, y_val,
              numTrimmed,numSkip,numK,resetAtrounds,easymode,
              numIter, numRounds, lenK,
              printStep=200, valStep=5)

trainer.evalMe(x_test,y_test,brickSize,numTrimmed,numSkip,lenK)
