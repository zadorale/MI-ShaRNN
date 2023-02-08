# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import numpy as np
import os
import sys
import microsoft_utils as utils
import SRNN_trainer
from microsoft_rnn import SRNN2

class ShaMIRNNTrainer:

    def __init__(self, srnnObj, learningRate,params, lossType='l2', device = None):
        '''
        A simple trainer for SRNN+MIRNN
        '''
        self.srnnObj = srnnObj
        self.__lR = learningRate
        self.params = params
        if device is None:
            self.device = "cpu"
        else:
            self.device = device
        self.trainer = SRNN_trainer.SRNNTrainer(self.srnnObj,self.__lR,
                                                lossType=lossType, device=self.device)

    def tobagform(self,x,bag_size):
        # input tensor of shape [datasize,numclasses]
        # or of [datasize]
        x = np.expand_dims(x,1)
        x = np.split(x,bag_size,0)
        x = np.concatenate(x,1)
        #output shape [bagamnt,bagsize,numClases]
        #or of [bagamnt,bagsize]
        return x

    def unBag(self,x):
        #output shape [bagamnt,bagsize,numClases]
        x = np.split(x,x.shape[1],1)
        x = np.concatenate(x,0)
        x = np.squeeze(x)
        #output shape [datasize,numClases]
        return x

    def slicer_MIRNN(self,x,y,lenS,skipS):
        """
        :param x: Tensor of shape [seq length, batch size, input dimension]
        :param y: Tensor of shape [batch size, num classes]
        :return:
        """
        xlis = list()
        ylis = list()
        bag_size = 0
        for i in range(0,(int(x.shape[0])-lenS+1),skipS):
            xlis.append(x[i:i+lenS])
            ylis.append(y)
            bag_size = bag_size+1
        xret = np.concatenate(xlis,1)
        yret = np.concatenate(ylis,0)
        return xret ,yret, bag_size

    def torch_slicer_MIRNN(self,x,y,lenS,skipS):
        """
        :param x: Tensor of shape [seq length, batch size, input dimension]
        :param y: Tensor of shape [batch size, num classes]
        :return:
        """
        xlis = list()
        ylis = list()
        bag_size = 0
        x = torch.tensor(x).to(self.device).float()
        for i in range(0,(int(x.shape[0])-lenS+1),skipS):
            xlis.append(x[i:i+lenS])
            ylis.append(y)
            bag_size = bag_size+1
        xret = torch.cat(xlis,1)
        yret = np.concatenate(ylis,0)
        return xret ,yret, bag_size

    def saveMe(self,PATH,num):
        PATH = PATH +"mod"+str(num)+".pth"
        torch.save(self.srnnObj.state_dict(), PATH)
        return
    def loadMe(self,PATH,num):
        PATH = PATH +"mod"+str(num)+".pth"
        self.srnnObj.load_state_dict(torch.load(PATH))
        self.trainer.setModel(self.srnnObj)
        return

    def resetMe(self):
        self.srnnObj = SRNN2(self.params[0],self.params[1],self.params[2],self.params[3],
                             self.params[4]).to(self.device)
        self.trainer.setModel(self.srnnObj)
        return

    def train(self, brickSize, batchSize, epochs, x_train, x_val, y_train, y_val ,
              lenS,skipS,k,reset,easymode,
              numIter, numRounds, lenK,
              printStep=10, valStep=1):
        """
        x_train, x_val, y_train, y_val: The numpy array containing train and
            validation data. x data is assumed to in of shape [timeSteps,
            -1, featureDimension] while y should have shape [-1, numberLabels].
        """
        directory =os.getcwd()+"/usedfortrainingthemodel/"
        path = directory
        try:
            print("[0m[1;30;43m"+'creating path: '+"[0m"+path)
            os.mkdir(path)
        except OSError:
            print("[0m[1;30;43m"+'this path already exists: '+"[0m"+path)
        nummodel = 1
        if(torch.cuda.is_available()):
            torch.cuda.empty_cache()
        self.saveMe(directory,0)
        numClasses = y_train.shape[-1]
        ctens =np.argmax(y_train,1)
        cvals =np.argmax(y_val,1)
        x_train , y_train , bag_size= self.slicer_MIRNN(x_train,y_train,lenS,skipS)

        x_val , y_val ,_ = self.slicer_MIRNN(x_val,y_val,lenS,skipS)
        smax = torch.nn.Softmax(dim=1)

        curr_y = y_train

        for cround in range(numRounds):
            print("[0m[1;37;44m"+"ROUND: "+str(cround)+"[0m")
            valAccList= []
            if(reset):
                if(not easymode):
                    self.resetMe()
                else:
                    self.loadMe(directory,0)
            for citer in range(numIter):
                print("[0m[1;37;42m"+"ITER: "+str(citer)+"[0m")
                #train model
                self.trainer.train(brickSize, batchSize, epochs, x_train,
                                   x_val, curr_y, y_val,
                                   printStep=1000, valStep=5)
                #update stuff like a val list
                self.saveMe(directory,nummodel)

                lis_out = []
                for i in range(0,x_val.shape[1]-1,50):
                    with torch.no_grad():
                        out = self.srnnObj.forward(torch.tensor(x_val[:,i:i+50]).to(self.device).float(),
                                                   brickSize)
                        out = smax(out).cpu().numpy()
                        out = np.argmax(out,1)
                        lis_out.append(out)

                out = np.concatenate(lis_out)
                out = self.tobagform(out, bag_size)
                out = self.getBagPredictions(out,lenK,numClasses)
                correct = (out == cvals)
                acc = np.mean(correct)
                valAccList.append((acc,nummodel))
                nummodel = nummodel+1

            # choose best model from val list
            valAccList.sort(key=lambda x: x[0], reverse=True)
            print(valAccList)
            toload = valAccList[0][1]
            self.loadMe(directory,toload)
            lis_out = []
            for i in range(0,x_train.shape[1],50):
                with torch.no_grad():
                    out = self.srnnObj.forward(torch.tensor(x_train[:,i:i+50]).to(self.device).float(),
                                               brickSize)
                    lis_out.append(  smax(out).cpu().numpy())
            out = np.concatenate(lis_out)
            newY = self.policyTopK(self.tobagform(curr_y, bag_size),
                                   self.tobagform(out, bag_size), ctens,
                                    numClasses,lenK)
            curr_y = self.unBag(newY)
        lis_out = []
        for i in range(0,x_val.shape[1],50):
            with torch.no_grad():
                out = self.srnnObj.forward(torch.tensor(x_val[:,i:i+50]).to(self.device).float(),brickSize)
                out = smax(out).cpu().numpy()
                out = np.argmax(out,1)
                lis_out.append(out)
        out = np.concatenate(lis_out)
        out = self.tobagform(out, bag_size)
        out = self.getBagPredictions(out,lenK,numClasses)
        correct = (out == cvals)
        print(str(correct.sum())+"correct of of"+str(cvals.shape[0]))
        acc = np.mean(correct)
        print("[0m[1;37;42m"+"FINAL SCORE FOR VALIDATION: " +"[0m"+ str(acc))
        return

    def evalMe(self,x_test,y_test,brickSize,
              lenS,skipS,lenK):
        numClasses = y_test.shape[-1]
        ctes =np.argmax(y_test,1)
        smax = torch.nn.Softmax(dim=1)
        x_test , y_test , bag_size= self.torch_slicer_MIRNN(x_test,y_test,lenS,skipS)
        lis_out = []
        for i in range(0,x_test.shape[1]-1,50):
            with torch.no_grad():
                out = self.srnnObj.forward(x_test[:,i:i+50].to(self.device).float(),
                                           brickSize)
                out = smax(out).cpu().numpy()
                out = np.argmax(out,1)
                lis_out.append(out)
        out = np.concatenate(lis_out)
        out = self.tobagform(out, bag_size)
        out = self.getBagPredictions(out,lenK,numClasses)
        correct = (out == ctes)
        acc = np.mean(correct)
        print("[0m[1;37;42m"+"SCORE FOR EVAL: " +"[0m"+ str(acc))
        return


    def getBagPredictions(self, Y_predicted, minSubsequenceLen = 4,
                          numClass=2, redirFile = None):
        '''
        Returns bag level predictions given instance level predictions.
        A bag is considered to belong to a non-zero class if
        minSubsequenceLen is satisfied. Otherwise, it is assumed
        to belong to class 0. class 0 is negative by default. If
        minSubsequenceLen is satisfied by multiple classes, the smaller of the
        two is returned
        Y_predicted is the predicted instance level results
        [-1, numsubinstance]
        Y True is the correct instance level label
        [-1, numsubinstance]
        '''
        assert(Y_predicted.ndim == 2)
        scoreList = []
        for x in range(1, numClass):
            scores = self.__getLengthScores(Y_predicted, val=x)
            length = np.max(scores, axis=1)
            scoreList.append(length)
        scoreList = np.array(scoreList)
        scoreList = scoreList.T
        assert(scoreList.ndim == 2)
        assert(scoreList.shape[0] == Y_predicted.shape[0])
        assert(scoreList.shape[1] == numClass - 1)
        length = np.max(scoreList, axis=1)
        assert(length.ndim == 1)
        assert(length.shape[0] == Y_predicted.shape[0])
        predictionIndex = (length >= minSubsequenceLen)
        prediction = np.zeros((Y_predicted.shape[0]))
        labels = np.argmax(scoreList, axis=1) + 1
        prediction[predictionIndex] = labels[predictionIndex]
        return prediction.astype(int)

    def __getLengthScores(self, Y_predicted, val=1):
        '''
        Returns an matrix which contains the length of the longest positive
        subsequence of val ending at that index.
        Y_predicted: [-1, numSubinstance] Is the instance level class
            labels.
        '''
        scores = np.zeros(Y_predicted.shape)
        for i, bag in enumerate(Y_predicted):
            for j, instance in enumerate(bag):
                prev = 0
                if j > 0:
                    prev = scores[i, j-1]
                if instance == val:
                    scores[i, j] = prev + 1
                else:
                    scores[i, j] = 0
        return scores

    def policyTopK(self, currentY, softmaxOut, bagLabel, numClasses, k=1):
        '''
        currentY: [-1, numsubinstance, numClass]
        softmaxOut: [-1, numsubinstance, numClass]
        bagLabel [-1]
        k: minimum length of continuous non-zero examples
        Algorithm:
            For each bag:
                1. Find the longest continuous subsequence of a label.
                2. If this label is the same as the bagLabel, and if the length
                of the subsequence is at least k:
                    2.1 Set the label of these instances as the bagLabel.
                    2.2 Set all other labels as 0
        '''
        assert currentY.ndim == 3
        assert k <= currentY.shape[1]
        assert k > 0
        # predicted label for each instance is max of softmax
        predictedLabels = np.argmax(softmaxOut, axis=2)
        scoreList = []
        # classScores[i] is a 2d array where a[j,k] is the longest
        # string of consecutive class labels i in bag j ending at instance k
        classScores = [-1]
        for i in range(1, numClasses):
            scores = self.__getLengthScores(predictedLabels, val=i)
            classScores.append(scores)
            length = np.max(scores, axis=1)
            scoreList.append(length)
        scoreList = np.array(scoreList)
        scoreList = scoreList.T
        # longestContinuousClass[i] is the class label having
        # longest substring in bag i
        longestContinuousClass = np.argmax(scoreList, axis=1) + 1
        # longestContinuousClassLength[i] is length of
        # longest class substring in bag i
        longestContinuousClassLength = np.max(scoreList, axis=1)
        assert longestContinuousClass.ndim == 1
        assert longestContinuousClass.shape[0] == bagLabel.shape[0]
        assert longestContinuousClassLength.ndim == 1
        assert longestContinuousClassLength.shape[0] == bagLabel.shape[0]
        newY = np.array(currentY)
        index = (bagLabel != 0)
        indexList = np.where(index)[0]
        # iterate through all non-zero bags
        for i in indexList:
            # longest continuous class for this bag
            lcc = longestContinuousClass[i]
            # length of longest continuous class for this bag
            lccl = int(longestContinuousClassLength[i])
            # if bagLabel is not the same as longest continuous
            # class, don't update
            if lcc != bagLabel[i]:
                continue
            # we check for longest string to be at least k
            if lccl < k:
                continue
            lengths = classScores[lcc][i]
            assert np.max(lengths) == lccl
            possibleCandidates = np.where(lengths == lccl)[0]
            # stores (candidateIndex, sum of probabilities
            # over window for this index) pairs
            sumProbsAcrossLongest = {}
            for candidate in possibleCandidates:
                sumProbsAcrossLongest[candidate] = 0.0
                # sum the probabilities over the continuous substring
                for j in range(0, lccl):
                    sumProbsAcrossLongest[candidate] += softmaxOut[i, candidate-j, lcc]
            # we want only the one with maximum sum of
            # probabilities; sort dict by value
            sortedProbs = sorted(sumProbsAcrossLongest.items(),key=lambda x: x[1], reverse=True)
            bestCandidate = sortedProbs[0][0]
            # apart from (bestCanditate-lcc,bestCandidate] label
            # everything else as 0
            newY[i, :, :] = 0
            newY[i, :, 0] = 1
            newY[i, bestCandidate-lccl+1:bestCandidate+1, 0] = 0
            newY[i, bestCandidate-lccl+1:bestCandidate+1, lcc] = 1
        return newY