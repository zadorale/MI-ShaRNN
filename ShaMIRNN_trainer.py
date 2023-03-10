# Copyright (c) Microsoft Corporation. All rights reserved.abs
# Licensed under the MIT license.

import torch
import numpy as np
import os
import sys
import microsoft_utils as utils

class ShaMIRNNTrainer:

    def __init__(self, srnnObj, learningRate, lossType='l2', device = None):
        '''
        A simple trainer for SRNN+MIRNN
        '''
        self.srnnObj = srnnObj
        self.__lR = learningRate
        self.lossType = lossType
        self.optimizer = self.__optimizer()
        self.lossCriterion = None
        assert lossType in ['l2', 'xentropy']
        if lossType == 'l2':
            self.lossCriterion = torch.nn.MSELoss()
            print("Using L2 (MSE) loss")
        else :
            self.lossCriterion = torch.nn.CrossEntropyLoss()
            print("Using x-entropy loss")

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

    def __optimizer(self):
        optimizer = torch.optim.Adam(self.srnnObj.parameters(),
                                     lr=self.__lR)
        return optimizer

    def loss(self, logits, labels_or_target):
        labels = labels_or_target
        assert len(logits) == len(labels)
        assert len(labels.shape) == 2
        assert len(logits.shape) == 2
        if self.lossType == 'xentropy':
            _, labels = torch.max(labels, dim=1)
            assert len(labels.shape)== 1
        loss = self.lossCriterion(logits, labels)
        return loss

    def accMI(self, logits, labels,og_batchsize):
        sumvec = torch.zeros(og_batchsize,logits.shape[1]).to(self.device)
        for i in range(0,int(logits.shape[0])-1,og_batchsize):
            sumvec = sumvec+ logits[i:i+og_batchsize]
        _, predictions = torch.max(sumvec, dim=1)
        acc , count = self.accuracy(predictions, labels)
        return acc, count

    def accuracy(self, predictions, labels):
        '''
        Returns accuracy and number of correct predictions.
        '''
        assert len(predictions.shape) == 1
        assert len(labels.shape) == 1
        if( not len(predictions) == len(labels)):
            print(str(predictions.shape)+"   "+str(labels.shape))
        assert len(predictions) == len(labels)
        correct = (predictions == labels).double()
        numCorrect = torch.sum(correct)
        acc = torch.mean(correct)
        return acc, numCorrect

    def slicer_MIRNN(self,x,y,lenS,skipS):
        """
        :param x: Tensor of shape [seq length, batch size, input dimension]
        :param y: Tensor of shape [batch size, num classes]
        :return:
        """
        xlis = list()
        ylis = list()
        for i in range(0,(int(x.shape[0])-lenS+1),skipS):
            xlis.append(x[i:i+lenS])
            ylis.append(y)
        xret = torch.cat(xlis,1)
        yret = torch.cat(ylis,0)
        return xret ,yret

    def train(self, brickSize, batchSize, epochs, x_train, x_val, y_train, y_val ,
              lenS,skipS,k,
              printStep=10, valStep=1):
        '''
        Performs training of SRNN.
        batchSize: Batch size per update
        epochs : The number of epochs to run training for. One epoch is
            defined as one pass over the entire training data.
        x_train, x_val, y_train, y_val: The numpy array containing train and
            validation data. x data is assumed to in of shape [timeSteps,
            -1, featureDimension] while y should have shape [-1, numberLabels].
        printStep: Number of batches between echoing of loss and train accuracy.
        valStep: Number of epochs between evaluations on validation set.
        '''
        L = self.srnnObj.outputDim
        assert batchSize >= 1, 'Batch size should be positive integer'
        assert epochs >= 1, 'Total epochs should be positive integer'
        assert x_train.ndim == 3, 'Expected training data to be of rank 3'
        assert x_val.ndim == 3, 'Expected validation data to be of rank 3'
        assert y_train.ndim == 2, 'Expected training labels to be of rank 2'
        assert y_train.shape[1] == L, 'Expected y_train to be [-1, %d]' % L
        assert y_val.ndim == 2, 'Expected validation labels to be of rank 2'
        assert y_val.shape[1] == L, 'Expected y_val to be [-1, %d]' % L

        trainNumBatches = int(np.ceil((x_train.shape[1]) / batchSize))
        valNumBatches = int(np.ceil((x_val.shape[1]) / batchSize))
        x_train_batches = np.array_split(x_train, trainNumBatches, axis=1)
        y_train_batches = np.array_split(y_train, trainNumBatches)
        x_val_batches = np.array_split(x_val, valNumBatches, axis=1)
        y_val_batches = np.array_split(y_val, valNumBatches)

        for epoch in range(epochs):
            for i in range(len(x_train_batches)):
                x_batch, y_batch = x_train_batches[i], y_train_batches[i]
                x_batch = torch.Tensor(x_batch)
                y_batch = torch.Tensor(y_batch)
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch, y_batch = self.slicer_MIRNN(x_batch, y_batch,lenS,skipS)
                self.optimizer.zero_grad()
                logits = self.srnnObj.forward(x_batch, brickSize)
                loss = self.loss(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                _, predictions = torch.max(logits, dim=1)
                _, target = torch.max(y_batch, dim=1)
                acc, _ = self.accuracy(predictions, target)
                if i % printStep == 0:
                    print("Epoch %d batch %d loss %f acc %f" % (epoch, i, loss,
                                                               acc))
            # Perform validation set evaluation
            if (epoch + 1) % valStep == 0 or (epoch == epochs - 1):
                numCorrect = 0
                numCorrectPerBag = 0
                for i in range(len(x_val_batches)):
                    x_batch, y_batch = x_val_batches[i], y_val_batches[i]
                    x_batch = torch.Tensor(x_batch)
                    y_batch = torch.Tensor(y_batch)
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    og_batchsize = int(y_batch.shape[0])
                    _ , bagTraget = torch.max(y_batch, dim=1);
                    x_batch, y_batch = self.slicer_MIRNN(x_batch, y_batch,lenS,skipS)
                    poistsplit_batchsize = int(y_batch.shape[0])
                    logits = self.srnnObj.forward(x_batch, brickSize)
                    _, bagCount = self.accMI(logits, bagTraget, og_batchsize);
                    _, predictions = torch.max(logits, dim=1)
                    _, target = torch.max(y_batch, dim=1)
                    _, count = self.accuracy(predictions, target)
                    numCorrect += count * (og_batchsize/poistsplit_batchsize)
                    numCorrectPerBag += bagCount
                print("Validation accuracy per seq: %f" % (numCorrect / x_val.shape[1]))
                print("Validation accuracy per bag: %f" %
                      (numCorrectPerBag / x_val.shape[1]))