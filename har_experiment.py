from __future__ import print_function
import sys
import subprocess
import os
import torch
# import torch
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
myUtils.directory_create(myUtils.HAR)
try:
    myUtils.download_file(myUtils.HAR)
except:
    print("[0m[1;37;41m"+"DOWNLOAD FAILED , PLEASE DOWNLOAD MANUALY FROM:"+"[0m"+"https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"+ "[0m[1;37;41m"+"AND PLACE IN HAR FOLDER "+"[0m")


def runcommand(command, splitChar=' '):
    p = subprocess.Popen(command.split(splitChar), stdout=subprocess.PIPE)
    output, error = p.communicate()
    assert(p.returncode == 0), 'Command failed: %s' % command

cwd =myUtils.directory + "/HAR/"
if('extracted_this.txt' in os.listdir(cwd)):
    print("[0m[1;37;41m"+'FILES ALREADY EXTRACTED'+"[0m")
else:
    zipplace = cwd + "UCI HAR Dataset.zip"

    import zipfile
    with zipfile.ZipFile(zipplace, 'r') as zip_ref:
        zip_ref.extractall(cwd)
    exec(open('process_HAR.py').read())

    f = open(cwd+'extracted_this.txt', 'w')
    f.write("extracted")
    f.close()


subinstanceLen = 48
subinstanceStride = 16

print("[0m[1;37;42m"+'FEATURES EXTRACTED STARTING EXPERIMENT'+"[0m")


from microsoft_rnn import SRNN2
import ShaMIRNNtrainer2 as trainerP
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

from microsoft_helpermethods2 import *
sDir = myUtils.directory+ "/HAR/RAW"
x_train, y_train, x_test, y_test, x_val, y_val = loadData(sDir)
print("data shapes:")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

x_train = np.swapaxes(x_train, 0, 1)
x_val = np.swapaxes(x_val, 0, 1)
x_test = np.swapaxes(x_test, 0, 1)

y_train = np.concatenate((np.zeros((y_train.shape[0],1)),y_train),axis =1)
y_val = np.concatenate((np.zeros((y_val.shape[0],1)),y_val),axis =1)
y_test = np.concatenate((np.zeros((y_test.shape[0],1)),y_test),axis =1)

print("Train shape", x_train.shape, y_train.shape)
print("Val shape", x_val.shape, y_val.shape)
print("Test shape", x_test.shape, y_test.shape)

numTimeSteps = x_train.shape[0]
numInput = x_train.shape[-1]
numClasses = y_train.shape[1]
subinstanceLen = 48
subinstanceStride = 16

# Network Parameters
brickSize = 16
hiddenDim0 = 32
hiddenDim1 = 8
cellType = 'LSTM'
learningRate = 0.001
batchSize = 32
epochs = 2

numK = 1
lenK = numK
numTrimmed = 48
numSkip = 16

numIter = 4
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


