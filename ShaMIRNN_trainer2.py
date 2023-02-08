# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import numpy as np
import os
import sys
import microsoft_utils as utils
import SRNN_trainer4

class ShaMIRNNTrainer:

    def __init__(self, srnnObj, learningRate, lossType='l2', device = None):
        '''
        A simple trainer for SRNN+MIRNN
        '''
        self.srnnObj = srnnObj
        self.__lR = learningRate

        if device is None:
            self.device = "cpu"
        else:
            self.device = device
        self.trainer = SRNN_trainer4.SRNNTrainer(self.srnnObj,self.__lR,
                                                lossType=lossType, device=self.device)

    def tobagform(x,bag_size):
        # input tensor of shape [datasize,numclasses]
        # or of [datasize]
        x = np.exapnd_dims(x,1)
        x = np.split(x,bag_size,0)
        x = np.concatenate(x,1)
        #output shape [bagamnt,bagsize,numClases]
        #or of [bagamnt,bagsize]
        return x

    def unBag(x):
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

    def train(self, brickSize, batchSize, epochs, x_train, x_val, y_train, y_val ,
              lenS,skipS,k,
              numIter, numRounds, lenK,
              printStep=10, valStep=1):
        """
        x_train, x_val, y_train, y_val: The numpy array containing train and
            validation data. x data is assumed to in of shape [timeSteps,
            -1, featureDimension] while y should have shape [-1, numberLabels].
        """
        numClasses = y_train.shape[-1]
        dat_size = x_train.shape[0]
        x_t , y_t , bag_size= self.slicer_MIRNN(x_train,y_train,lenS,skipS)
        x_v , y_v ,_ = self.slicer_MIRNN(x_val,y_val,lenS,skipS)
        smax = torch.nn.Softmax(dim=1)
        ctens =np.argmax(y_train,1)

        curr_y = y_t

        for cround in range(numRounds):
            print("[0m[1;37;44m"+"ROUND: "+str(cround)+"[0m")
            valAccList, globalStepList = [], []

            for citer in range(numIter):
                #train model
                self.trainer.easytrain(brickSize, batchSize, epochs, x_t,
                                   x_v, curr_y, y_v,
                                   printStep=1000, valStep=1)
                #TO DO-update stuff like a val list

            #TO DO- choose best model from val list


            with torch.no_grad():
                out = smax(self.srnnObj(torch.tensor(x_t).to(self.device).float(),k)).cpu().numpy()
                out = self.tobagform(out, bag_size)

            #out= [np.array(out[i][0]) for i in range(len(out))]
            #out = np.concatenate(out)[:, :, -1, :]
            newY = self.policyTopK(self.tobagform(curr_y, bag_size), out, ctens,
                                    numClasses)
            curr_y = self.unBag(newY)

        with torch.no_grad():
            out = smax(self.srnnObj(torch.tensor(x_t).to(self.device),k)).cpu().numpy()
            out = np.argmax(out,1)
            pred = self.tobagform(out, bag_size)

        pred = self.getBagPredictions(pred,lenK,numClasses)
        correct = (pred == ctens).double()
        acc = torch.mean(correct)
        print("AAAAAAAAAAAAAAAAA" + str(acc))
        return




    def getInstancePredictions(self, x, y, earlyPolicy, batchSize=1024,
                               feedDict=None, **kwargs):

        '''
        Returns instance level predictions for data (x, y).
        Takes the softmax outputs from the joint trained model and, applies
        earlyPolicy() on each instance and returns the instance level
        prediction as well as the step at which this prediction was made.
        softmaxOut: [-1, numSubinstance, numTimeSteps, numClass]
        earlyPolicy: callable,
            def earlyPolicy(subinstacePrediction):
                subinstacePrediction: [numTimeSteps, numClass]
                ...
                return predictedClass, predictedStep
        returns: predictions, predictionStep
            predictions: [-1, numSubinstance]
            predictionStep: [-1, numSubinstance]
        '''
        opList = self._emiTrainer.softmaxPredictions
        if 'keep_prob' in kwargs:
            assert kwargs['keep_prob'] == 1, 'Keep prob should be 1.0'
        smxOut = self.runOps(opList, x, y, batchSize, feedDict=feedDict,
                             **kwargs)
        softmaxOut = np.concatenate(smxOut, axis=0)
        assert softmaxOut.ndim == 4
        numSubinstance, numTimeSteps, numClass = softmaxOut.shape[1:]
        softmaxOutFlat = np.reshape(softmaxOut, [-1, numTimeSteps, numClass])
        flatLen = len(softmaxOutFlat)
        predictions = np.zeros(flatLen)
        predictionStep = np.zeros(flatLen)
        for i, instance in enumerate(softmaxOutFlat):
            # instance is [numTimeSteps, numClass]
            assert instance.ndim == 2
            assert instance.shape[0] == numTimeSteps
            assert instance.shape[1] == numClass
            predictedClass, predictedStep = earlyPolicy(instance, **kwargs)
            predictions[i] = predictedClass
            predictionStep[i] = predictedStep
        predictions = np.reshape(predictions, [-1, numSubinstance])
        predictionStep = np.reshape(predictionStep, [-1, numSubinstance])
        return predictions, predictionStep

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