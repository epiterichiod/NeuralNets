# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:08:54 2021

@author: epite
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow import keras
from genData import *
from polarcodes import *

import csv 



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


CSVfilename = 'FC_NN_Log.csv'

codeWrdLen = 8
msgLen = 4
numMsg = 30    *1000 # in thousands
desSnr = 20   #in dB


epic = 20   #the number of times it generates new data, after each set of epochs (yes, I do think I'm funny ;)  )
epochs = 35
batchSize = 2048 #high batch sizes have been running faster on my GPU and it seems they can lead to a more generalized result 

numTst = 100 #number of test messages to run after the net has been trained. Currently spits(prints) out TONS of data at the end FYI
testSNR = 20 #The SNR above sets the encoder/decoder function and is the training rate. This SNR is what the net is tested to below




loss = 'binary_crossentropy'

layer1Nodes = 1024
layer2Nodes = 1024
layer3Nodes = 0
layer4Nodes = 0
layer5Nodes = 0
layer6Nodes = 0
layer7Nodes = 0
layer8Nodes = 0


initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=10.0)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(codeWrdLen, activation='sigmoid', kernel_regularizer = l2(0.0005), kernel_initializer = initializer),
  tf.keras.layers.Dense(layer1Nodes, activation= 'sigmoid', kernel_regularizer = l2(0.0005),kernel_initializer = initializer),
  tf.keras.layers.Dense(layer2Nodes, activation= 'sigmoid', kernel_regularizer = l2(0.0005),kernel_initializer = initializer),
  tf.keras.layers.Dense(codeWrdLen, activation='sigmoid')
])


callback1 = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience=12) #added max and baseline
snr = desSnr
opt = keras.optimizers.Adam(learning_rate = 0.00035)

model.compile(optimizer=opt,
              #loss='binary_crossentropy',
              #loss = 'mse',
              loss = loss,
              metrics=[ tf.keras.metrics.BinaryAccuracy(),'accuracy'])

for i in range(0, epic):
    
    print("epic #: ",i)
    print('\n')
    #txCW, ogCW =  genChanDataRetCW(codeWrdLen, msgLen, numMsg,snr) #runs on the other PyPolar library, worked, but need to switch everything over. Can't decode predicted CWs
    
    #This generates a new batch of data at each epoch. My poor 12 GB ram can't hold all of the training data required for large
    #message lengths. To get around this, we change up the training data every so often. keeps things fresh even on the smaller 
    #message lengths. 
    txCW, ogCW, ogMsg = genChanDataRetAllog(codeWrdLen,msgLen, numMsg, desSnr)
    
    #trains the model on the data just generated. 
    hist = model.fit(txCW, ogCW, epochs=epochs, batch_size = batchSize, callbacks =[callback1]  )
    
    
    
    
    
    
    
    
    
    
    
################################################################################################

#Test

################################################################################################
print('\n','\n','\n','\n','\n')
#Again, generating data, Test Data this time:
txCW, ogCW, ogMsg = genChanDataRetAllog(codeWrdLen,msgLen,numTst, testSNR)

#shoves the new data through the NN model. 
predData = model.predict(txCW) #Pass in our test data, forward propogate and get predicted answers


#This is kind of a curiousity check, it checks the predicted values against the original untransmitted CW
codeWordBitMismatch = checkBin(predData,ogCW) #checks binary error between the codewords 

#This decodes the original transmitted CodeWords
#NOTE, txCW is in the "constellation" (+-) form, THIS IS NECCESARY FOR DECODING
decOgCW = decodeBlob(txCW,codeWrdLen,msgLen,desSnr)

#This checks the biterrors on the "received" values. No NN involved. Used as a baseline
noNNBitErrs = checkBin(decOgCW,ogMsg)
noNNpercErr = noNNBitErrs/(numTst*msgLen)
print("^checkBin on OGcw and OGmsg",'\n', "The total bit errrors for the TX w/o NN: ",noNNBitErrs," that's ~",noNNBitErrs/(numTst*msgLen),'\n')



#This is the important one 
#We first decoede the NN pred data.First it is fed into the binary to constellation form function 1/0 -> +- form
#then that is pushed through decode, Then we check the bit errors agianst the original message.
decPred = decodeBlob(binToConst(predData),codeWrdLen,msgLen,desSnr)
NNBitErrs = checkBin(decPred,ogMsg)
NNpercErr = NNBitErrs/(numTst*msgLen)
print("^checkBin on decoded PredData and OGmsg",'\n', "The total bit errrors for the TX w/ NN: ",NNBitErrs," that's ~",NNBitErrs/(numTst*msgLen),'\n')





csvToWrite = [str(1-NNpercErr), str(1-noNNpercErr), (codeWordBitMismatch/(codeWrdLen*numTst)), desSnr, testSNR, codeWrdLen, msgLen, (msgLen/codeWrdLen), numMsg, batchSize, epochs, epic, loss, layer1Nodes,
              layer2Nodes, layer3Nodes, layer4Nodes, layer5Nodes, layer6Nodes, layer7Nodes, layer8Nodes]


addRowToCSV(CSVfilename, csvToWrite)










