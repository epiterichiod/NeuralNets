from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes import FastSSCPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,)
import numpy as np
import random
import csv 
from csv import writer
########################################

from polarcodes import *
    

def addRowToCSV(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
        
        
        
        

def genData( codeWordLength, dataLength, numMessages,):
    
    N = codeWordLength
    K = dataLength
    design_snr = 2
    messages = numMessages
    # SNR in [.0, .5, ..., 4.5, 5]
    snr_range = [i / 2 for i in range(11)]


    encDataOut = np.zeros([messages,N])
    OGdataOut = np.zeros([messages,K])

    codec = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)


    for i in range(0,messages):
        
        msg = generate_binary_message(size=K)
        encodedMsg = codec.encode(msg)
        encDataOut[i] = encodedMsg != 0
        OGdataOut[i] = msg != 0
        
    return encDataOut, OGdataOut;


def genChanData(codeWordLen, msgLen,numMsg, snr):
    
    encDataOut = np.zeros([numMsg,codeWordLen]);
    ogMsg = np.zeros([numMsg,msgLen])

    # initialise polar code
    myPC = PolarCode(codeWordLen, msgLen)
    myPC.construction_type = 'bb'
    
    # mothercode construction
    design_SNR  = snr
    Construct(myPC, design_SNR)
    #print(myPC, "\n\n")
    
    
    for i in range(0, numMsg):

        # set message
        my_message = np.random.randint(2, size=myPC.K)
        myPC.set_message(my_message)
        #print("The message is:", my_message)
    
        Encode(myPC)    
        AWGN(myPC, design_SNR)
    
        encDataOut[i] = myPC.likelihoods
        ogMsg[i] = my_message;
    
    
    return encDataOut, ogMsg

def genChanDataRetCW(codeWordLen, msgLen,numMsg, snr):
    
    txCodeWord = np.zeros([numMsg,codeWordLen]);
    ogCW = np.zeros([numMsg,codeWordLen])

    # initialise polar code
    myPC = PolarCode(codeWordLen, msgLen)
    myPC.construction_type = 'bb'
    
    # mothercode construction
    design_SNR  = snr
    Construct(myPC, design_SNR)
    #print(myPC, "\n\n")
    
    
    for i in range(0, numMsg):

        # set message
        my_message = np.random.randint(2, size=myPC.K)
        myPC.set_message(my_message)
        #print("The message is:", my_message)
    
        Encode(myPC)    
        AWGN(myPC, design_SNR)
    
        txCodeWord[i] = myPC.likelihoods
        ogCW[i] = myPC.get_codeword()
    
    
    return txCodeWord, ogCW
    
def genChanDataRetAll(codeWordLen, msgLen,numMsg, snr):
    
    txCodeWord = np.zeros([numMsg,codeWordLen]);
    ogCW = np.zeros([numMsg,codeWordLen])
    ogMsg = np.zeros([numMsg,msgLen])
    # initialise polar code
    myPC = PolarCode(codeWordLen, msgLen)
    myPC.construction_type = 'bb'
    
    # mothercode construction
    design_SNR  = snr
    Construct(myPC, design_SNR)
    #print(myPC, "\n\n")
    
    
    for i in range(0, numMsg):

        # set message
        my_message = np.random.randint(2, size=myPC.K)
        myPC.set_message(my_message)
        #print("The message is:", my_message)
    
        Encode(myPC)    
        AWGN(myPC, design_SNR)
    
        txCodeWord[i] = myPC.likelihoods
        ogCW[i] = myPC.get_codeword()
        ogMsg[i] = my_message
    
    
    return txCodeWord, ogCW, myPC, ogMsg

def genChanDataRetAllog(codeWordLen, msgLen,numMsg, snr):
    
    N = codeWordLen
    K = msgLen
    design_snr = snr
    messages = numMsg


    encDataOut = np.zeros([messages,N])
    OGdataOut = np.zeros([messages,K])
    bpskCW = np.zeros([messages,N])

    codec = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)


    for i in range(0,messages):
        
        msg = generate_binary_message(size=K)
        encodedMsg = codec.encode(msg)
        encDataOut[i] = encodedMsg 
        OGdataOut[i] = msg 
        bpskCW[i] = bpsk.transmit(message = encodedMsg, snr_db = snr)
    
    return bpskCW, encDataOut, OGdataOut
    
    
    


def checkBin( yPredOG, yRealOG):
    
    yPred = np.copy(yPredOG)
    yReal = np.copy(yRealOG)
    
    totErr = 0
    
    for i in range(0,yReal.shape[0]):
        for j in range(0,yReal.shape[1]):
            
            yPred[i][j] = round(yPred[i][j])
            
        delta = yPred[i]-yReal[i]
        delta = delta*delta
        #print("Vector #",i," had ", sum(delta), "Bit errors.")
        totErr += sum(delta)
    print("Total number of Bit errors: ", totErr)
    
    return totErr
    
    
def checkBinConst( yPredOG, yRealOG):
    
    yPred = np.copy(yPredOG)
    yReal = np.copy(yRealOG)
    
    totErr = 0
    
    for i in range(0,yReal.shape[0]):
        vecErr = 0
        for j in range(0,yReal.shape[1]):
            if(yPred[i][j]>0):
                yPred[i][j] = 1
            if(yPred[i][j]<0):
                yPred[i][j] = -1   
                
            if(yReal[i][j]>0):
                yReal[i][j] = 1
            if(yReal[i][j]<0):
                yReal[i][j] = -1                   
                
                
        delta = yPred[i]+yReal[i]
        
        for k in range(0,delta.size):
            if(delta[k] == 0):
                vecErr += 1
        print("Vector #",i," had ", vecErr, "Bit errors.")
        totErr += vecErr
    print("Total number of Bit errors: ", totErr)


def genBitErrs(goodArray,numErrs):
    
    #NumErrs = number of errors to input into each vector
    if(numErrs > goodArray.shape[1]):
        print("The number of errors per vector must be SMALLER than the number of elements in the vector")
        return

    badArray = goodArray.copy()

        
    for i in range(0,goodArray.shape[0]):        
        errsToInsert = numErrs
        
        while(errsToInsert):
            errsToInsert -= 1
            curIdx = random.randint(0,goodArray.shape[1]-1)
            
            if(goodArray[i][curIdx] < 0.5):
                badArray[i][curIdx] = 1    
            else:
                badArray[i][curIdx] = 0

    return badArray
            

def decodeBlob( CWarray, codeWordLen, msgLen, snr):
    
    
    #This requires CWarray to be in constellation form ie: +-vals not Bin.
    
    N = codeWordLen
    K = msgLen
    design_snr = snr


    decDataOut = np.zeros([CWarray.shape[0],K])

    codec = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
    for i in range(0,CWarray.shape[0]):
        decDataOut[i] = codec.decode(CWarray[i])
            
    return decDataOut
        


def binToConst(CWarrayOG):
    
    CWarray = np.copy(CWarrayOG)
    
    for i in range(0,CWarray.shape[0]):
        for j in range(0,CWarray.shape[1]):
            if(CWarray[i][j]>0.5):
                CWarray[i][j] = -1
            else:
                CWarray[i][j] = 1    
    return CWarray                
                
    
    
        
    
    
    
    
