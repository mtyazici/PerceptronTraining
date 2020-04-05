#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import pylab
from mpl_toolkits.mplot3d import Axes3D

def dataGeneration2(numA,numB):
    nA = numA
    nB = numB
    mA = [1.0,0.3]
    sigmaA = 0.2
    mB = [0, -0.1]
    sigmaB=0.3
    classALine1 = np.concatenate((np.random.normal(-mA[0], sigmaA, [1, round(0.5*numA)]),np.random.normal(mA[0], sigmaA, [1,round(0.5*numA)])),axis=1)
    classALine2 = np.random.normal(mA[1],sigmaA,[1,numA])
    classA = np.concatenate((classALine1,classALine2),axis=0)
    classBLine1 = np.random.normal(mB[0],sigmaB,[1,numB])
    classBLine2 = np.random.normal(mB[1],sigmaB,[1,numB])
    classB = np.concatenate((classBLine1,classBLine2),axis=0)
    return classA, classB


def twoLayersPerceptron(nN,nA,nB,eta,classA,classB,method,epochs,perValA= 0.0,perValB = 0.0,alpha = 0.9):
    def phi(x):
        return (2/(1+np.exp(-x))-1)
    def phiPrime(x):
        phiValue = phi(x)
        return ((1+phiValue)*(1-phiValue))/2
    W1 = np.random.randn(nN,3)
    W2 = np.random.randn(1,nN+1)
    dW1 = np.zeros([nN,3])
    dW2=  np.zeros([1,nN+1])
    biasA = np.ones([1, nA])
    biasB = np.ones([1, nB])
    
    
    permutation = np.random.permutation(classA.shape[1])
    
    classA = classA[:, permutation]
    classB = classB[:, permutation]

    
    classABiased = np.concatenate((classA, biasA), axis=0)  #3x100
    classABiasedT = classABiased[:,int(nA*perValA):]        #3x80
    classABiasedV = classABiased[:,:int(nA*perValA)]        #3x20
    targetTraA = np.ones([1, int(nA*(1-perValA))])          #1x80
    targetValA = np.ones([1, int(nA*perValA)])              #1x20
    
    classBBiased = np.concatenate((classB, biasB), axis=0)
    classBBiasedT = classBBiased[:,int(nB*perValB):]
    classBBiasedV = classBBiased[:,:int(nB*perValB)]
    targetTraB = -np.ones([1, int(nB*(1-perValB))])
    targetValB = -np.ones([1, int(nB*perValB)])
    
    epoch = np.concatenate((classABiasedT, classBBiasedT), axis=1)  #3x170
    val = np.concatenate((classABiasedV,classBBiasedV), axis=1)     #3x30
    target = np.concatenate((targetTraA, targetTraB), axis=1)       #1x170
    targetVal = np.concatenate((targetValA, targetValB),axis =1)    #1x30
    #Randomize the learning data and targets (not necessary when we do the batch algorithm)
    
    #epoch = epoch[:, permutation]
    #target = target[:, permutation]

    trainingAccuracy = []
    trainingError = []
    validationAccuracy = []
    validationError = []
    if method == 'batch':
        for k in range(epochs):
            hStar = W1@epoch
            h = phi(hStar)
            biasH = np.ones([1,epoch.shape[1]])
            h = np.concatenate((h,biasH), axis=0)
            yStar = W2@h
            y = phi(yStar)
            
            hVal =phi(W1@val)
            biasHVal = np.ones([1,val.shape[1]])
            hVal = np.concatenate((hVal,biasHVal), axis=0)
            yVal= phi(W2@hVal)
            
            deltaY = (y-target)*phiPrime(yStar)
            deltaH = (W2.T@deltaY)[:nN,:] * phiPrime(hStar)
            dW1 = dW1*alpha - (deltaH@epoch.T)
            dW2 = dW2*alpha - (deltaY@h.T)
            W1 += dW1*eta
            W2 += dW2*eta
           
            #Evaluating how's the learning
            wellClassified = 0
            for i in range(int(nA*(1-perValA)+nB*(1-perValB))):
                if yStar[0, i] > 0:
                    if target[0, i] == 1:
                        wellClassified += 1
                else:
                    if target[0, i] == -1:
                        wellClassified += 1     
            trainingAccuracy.append(wellClassified/(nA*(1-perValA)+nB*(1-perValB)))
            err = ((y-target)@(y-target).T)/2
            trainingError.append(err.squeeze().squeeze())
            
            
            if(perValA+perValB > 0):
                wellClassified = 0
                for i in range(int(nA*perValA+nB*perValB)):
                    if yVal[0, i] > 0:
                        if targetVal[0, i] == 1:
                            wellClassified += 1
                    else:
                        if targetVal[0, i] == -1:
                            wellClassified += 1 
                validationAccuracy.append(wellClassified/(nA*perValA+nB*perValB))
                err = ((yVal-targetVal)@(yVal-targetVal).T)/2
                validationError.append(err.squeeze().squeeze())
            
    if method == 'sequential':
        y= []
        yVal= []
        for k in range(epochs):
            yStar = []
            for i in range(int(nA*(1-perValA)+nB*(1-perValB))):
                hStar = np.dot(W1, epoch[:,i]).reshape((W1.shape[0],1))
                h = phi(hStar)
                biasH = np.ones((1,1))
                h = np.concatenate((h,biasH), axis=0)
                yStar = np.dot(W2, h)
                y = phi(yStar)
                

                #hValStar = W1@val[:,i].reshape((val.shape[0],1)) 
                #hVal =phi(hValStar)
                #biasHVal = np.ones((1,1))
                #hVal = np.concatenate((hVal,biasHVal), axis=0)
                #yVal= phi(W2@hVal)
            
            
                deltaY = np.multiply(np.add(y, -target[0,i]), phiPrime(yStar))
                deltaH = np.multiply(np.dot(np.transpose(W2), deltaY)[:nN,:], phiPrime(hStar))
                dW1 = dW1*alpha - (deltaH@epoch[:,i].reshape((epoch.shape[0],1)).T)
                dW2 = dW2*alpha - (deltaY@h.T)
                W1 += dW1*eta
                W2 += dW2*eta
            
            hVal =phi(W1@val)
            biasHVal = np.ones([1,val.shape[1]])
            hVal = np.concatenate((hVal,biasHVal), axis=0)
            yVal= phi(W2@hVal)
            
            hStar = W1@epoch
            h = phi(hStar)
            biasH = np.ones([1,epoch.shape[1]])
            h = np.concatenate((h,biasH), axis=0)
            yStar = W2@h
            y = phi(yStar)
            
            #Evaluating how's the learning
            wellClassified = 0
            for i in range(int(nA*(1-perValA)+nB*(1-perValB))):
                if yStar[0, i] > 0:
                    if target[0, i] == 1:
                        wellClassified += 1
                else:
                    if target[0, i] == -1:
                        wellClassified += 1     
            trainingAccuracy.append(wellClassified/(nA*(1-perValA)+nB*(1-perValB)))
            err = ((y-target)@(y-target).T)/2
            trainingError.append(err.squeeze().squeeze())
            
            if(perValA+perValB > 0):
                wellClassified = 0
                for i in range(int(nA*perValA+nB*perValB)):
                    if yVal[0, i] > 0:
                        if targetVal[0, i] == 1:
                            wellClassified += 1
                    else:
                        if targetVal[0, i] == -1:
                            wellClassified += 1 
                validationAccuracy.append(wellClassified/(nA*perValA+nB*perValB))
                err = ((yVal-targetVal)@(yVal-targetVal).T)/2
                validationError.append(err.squeeze().squeeze())
            
            
    return trainingAccuracy, trainingError , validationAccuracy , validationError , W1,W2,classABiasedT[:2,:],classBBiasedT[:2,:],classABiasedV[:2,:],classBBiasedV[:2,:]



def NNOutput(inp,W1,W2):
    def phi(x):
        return (2/(1+np.exp(-x))-1)
    inp = np.concatenate((inp,np.ones((1,inp.shape[1]))),axis= 0)
    hStar = W1@inp
    h = phi(hStar)
    biasH = np.ones([1,inp.shape[1]])
    h = np.concatenate((h,biasH), axis=0)
    yStar = W2@h
    y = phi(yStar)
    return y

def findDecisionBoundary(W1,W2,classA,classB,ax1):
    epsilon = 0.009;
    x = np.arange(-2.0,2.0,0.005)
    y = np.arange(-2.0,2.0,0.005)
    lenn = len(x)
    l2 = lenn*lenn
    xx,yy = np.meshgrid(x,y)
    xx = xx.reshape((1,l2))
    yy = yy.reshape((1,l2))
    patterns = np.concatenate((xx,yy),axis = 0)
    output = NNOutput(patterns,W1,W2)
    dataX = []
    dataY = []
    for i in range(output.shape[1]):
        if(abs(output[0][i]) < epsilon ):
            dataX.append(xx[0][i])
            dataY.append(yy[0][i])
    ax1.plot(dataX,dataY,'yd',label='decision boundary')  
    ax1.plot(classA[0, :], classA[1, :], 'bo', label='Class A')
    ax1.plot(classB[0, :], classB[1, :], 'ro', label='Class B')
    ax1.legend()
         
        
def findDes():
    eta=0.0001
    method = 'sequential'
    epochs =  1000
    nOfNodes= 5
    Accuracy = []
    Error = []
    nA=100
    nB=100
    classA ,classB = dataGeneration2(nA,nB)
    tAc, tEr, vAc,vEr ,w1,w2,a,b,c,d = twoLayersPerceptron(nOfNodes, nA, nB, eta, classA, classB, method, epochs,0.1,0.1)
    ax1 = plt
    findDecisionBoundary(w1,w2,classA,classB,ax1)



def visualizeTrainedData(W1,W2,classA,classB,Accuracy,Error,epochs,title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5),constrained_layout=True)
    findDecisionBoundary(W1,W2,classA,classB,ax1)
   # fig.tight_layout()
    ax2.plot(np.arange(epochs),Accuracy, label='Accuracy over iterations', color= 'blue')
    ax3.plot(np.arange(epochs),Error, label='Error over iterations', color ='blue')
    ax2.legend()
    fig.suptitle(title)
    ax3.legend()
    plt.show()


def changeHiddenNodes_3_2_1_1():
    nA=100
    nB=100
    eta=0.0005
    method = 'batch'
    epochs =  1000
    classA, classB = dataGeneration2(nA,nB)
    nOfNodes= 15
    Accuracy = []
    Error = []
    for i in range(nOfNodes):
        tAc, tEr, vAc,vEr,w1,w2,a,b,c,d = twoLayersPerceptron(i+1, nA, nB, eta, classA, classB, method, epochs)
        Accuracy.append(tAc[-1])
        Error.append(tEr[-1])
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    ax1.plot(np.arange(nOfNodes)+1,Accuracy,color='red' ,label='Accuracy over Number of Nodes')
    ax2.plot(np.arange(nOfNodes)+1,Error,color='blue',label='Error over Number of Nodes')
    ax1.legend()
    ax2.legend()
    plt.show()     



def compareTrainingAndValidation_3_2_1_2_1():
    nA=100
    nB=100
    eta=0.0001
    method = 'batch'
    epochs =  1500
    classA, classB = dataGeneration2(nA,nB)
    nOfNodes= 5
    perValAB = [[0.25,0,25],[0.5,0.0],[0.0,0.5]]
    for i in range(len(perValAB)):
        tAc, tEr, vAc ,vEr ,w1,w2,a,b,c,d= twoLayersPerceptron(nOfNodes, nA, nB, eta, classA, classB, method , epochs, perValAB[i][0], perValAB[i][1])
        visualizeTrainedData(w1,w2,a,b,tAc,tEr,epochs,'Training')
        visualizeTrainedData(w1,w2,c,d,vAc,vEr,epochs, 'Validation')


def compareTrainingAndValidationWrtHiddenNodes_3_2_1_2_2():
    nA=100
    nB=100
    eta=0.001
    method = 'batch'
    epochs =  10
    classA, classB = dataGeneration2(nA,nB)
    nOfNodes= 5
    perValA = 0.1
    perValB= 0.2
    perValAB = [[0.25,0,25],[0.5,0.0],[0.0,0.5]]
    for i in range(3):
            for j in range(nOfNodes):
                tAc, tEr, vAc ,vEr = twoLayersPerceptron(nOfNodes, nA, nB, eta, classA, classB, method , epochs, perValAB[i][0], perValAB[i][1])
                visualizeTrainedData(classA,classB,tAc,tEr,epochs)
                visualizeTrainedData(classA,classB,vAc,vEr,epochs)



def compareTrainingAndValidationWrtBatchSequential_3_2_1_2_3():
    nA=100
    nB=100
    eta=0.001
    epochs =  10
    classA, classB = dataGeneration2(nA,nB)
    nOfNodes= 10
    perValA = 0.1
    perValB= 0.2
    perValAB = [[0.25,0,25],[0.5,0.0],[0.0,0.5]]
    for i in range(3):
        tAc, tEr, vAc ,vEr = twoLayersPerceptron(nOfNodes, nA, nB, eta, classA, classB, 'batch' , epochs, perValAB[i][0], perValAB[i][1])
        tAc, tEr, vAc ,vEr = twoLayersPerceptron(nOfNodes, nA, nB, eta, classA, classB, 'sequential' , epochs, perValAB[i][0], perValAB[i][1])
        visualizeTrainedData(classA,classB,tAc,tEr,epochs)
        visualizeTrainedData(classA,classB,vAc,vEr,epochs)



def autoEncoder(dimInput,nN,eta,method,epochs,alpha= 0.9):
    def phi(x):
        return (2/(1+np.exp(-x))-1)
    def phiPrime(x):
        phiValue = phi(x)
        return ((1+phiValue)*(1-phiValue))/2
    def createInput(dimInput):
        inp = -np.ones((dimInput,1))
        inp[0] = 1
        for i in range(dimInput-1):
            inp = np.concatenate((inp,-np.ones((dimInput,1))),axis=1    )
            inp[i+1][i+1] = 1
        return inp
    
    W1 = np.random.randn(nN,dimInput+1)
    W2 = np.random.randn(dimInput,nN+1)
    dW1 = np.zeros([nN,dimInput+1])
    dW2=  np.zeros([dimInput,nN+1])
    bias = np.ones([1, dimInput])
    inp = createInput(dimInput)
    target = inp
    epoch = np.concatenate((inp, bias), axis=0)  
    #Randomize the learning data and targets (not necessary when we do the batch algorithm)
    #permutation = np.random.permutation(epoch.shape[1])
    #epoch = epoch[:, permutation]
    #target = target[:, permutation]
    trainingError = []
    h = []
    output = []
    if method == 'batch':
        for k in range(epochs):
            hStar = W1@epoch
            h = phi(hStar)
            biasH = np.ones([1,epoch.shape[1]])
            h = np.concatenate((h,biasH), axis=0)
            yStar = W2@h
            y = phi(yStar)           
            output = y
            deltaY = (y-target)*phiPrime(yStar)
            deltaH = (W2.T@deltaY)[:nN,:] * phiPrime(hStar)
            dW1 = dW1*alpha - (deltaH@epoch.T)
            dW2 = dW2*alpha - (deltaY@h.T)
            W1 += dW1*eta
            W2 += dW2*eta
            
            #Evaluating how's the learning
            err = ((y-target).sum(axis = 0)@(y-target).sum(axis = 0).T)/2
            trainingError.append(err.squeeze().squeeze())
            
    if method == 'sequential':
        for k in range(epochs):
            wellClassified = 0
            # Randomize the learning data and targets (not necessary when we do the batch algorithm)
            permutation = np.random.permutation(epoch.shape[1])
            epoch = epoch[:, permutation]
            target = target[:, permutation]
            for i in range(nA+nB):
                hStar = np.dot(W1, epoch[:,i])
                h = phi(hStar)
                yStar = np.dot(W2, h)
                y = phi(yStar)
                deltaY = np.multiply(np.add(-y, target[0,i]), phiPrime(yStar))
                deltaH = np.multiply(np.dot(np.transpose(W2), deltaY), phiPrime(hStar))
                W1 += eta*np.dot(deltaH, np.transpose(epoch[:,i]))
                W2 += eta*deltaY*np.transpose(h)
            #Evaluating how's the learning 
            for i in range(nA+nB):
                if y > 0:
                    if target[0, i] == 1:
                        wellClassified += 1
                else:
                    if target[0, i] == -1:
                        wellClassified += 1
            Accuracy.append(wellClassified/(nA+nB))
            Error.append(((y-target)@(y-target).T)/2)
    return trainingError , output ,hStar



def printMatrixE(a):
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print(("%6.3f" %a[i,j]), end = '   ')
      print()
   print()   



def trainFunctionApproximate(patterns,targets,eta,nN,method,epochs,perVal =0.0,alpha=0.9):
    def phi(x):
        return (2/(1+np.exp(-x))-1)
    def phiPrime(x):
        phiValue = phi(x)
        return ((1+phiValue)*(1-phiValue))/2
    W1 = np.random.randn(nN,3)
    W2 = np.random.randn(1,nN+1)
    dW1 = np.zeros([nN,3])
    dW2=  np.zeros([1,nN+1])
    nOfPattern = patterns.shape[1]
    bias = np.ones([1,nOfPattern ])
    
    
    permutation = np.random.permutation(patterns.shape[1])
    patterns= patterns[:,permutation]
    targets= targets[:,permutation]
    
    patternsBiased = np.concatenate((patterns, bias), axis=0)  #3x100
    val = patternsBiased[:,:int(nOfPattern*perVal)]        #3x20
    epoch = patternsBiased[:,int(nOfPattern*perVal):]        #3x80
    targetVal = targets[:,:int(nOfPattern*perVal)] 
    target = targets[:,int(nOfPattern*perVal):]            #1x20
    
    #Randomize the learning data and targets (not necessary when we do the batch algorithm)
    
    
    trainingError = []
    validationError = []
    output = []
    if method == 'batch':
        for k in range(epochs):
            hStar = W1@epoch
            h = phi(hStar)
            biasH = np.ones([1,epoch.shape[1]])
            h = np.concatenate((h,biasH), axis=0)
            yStar = W2@h
            y = phi(yStar)
            
            output = y
            
            hVal =phi(W1@val)
            biasHVal = np.ones([1,val.shape[1]])
            hVal = np.concatenate((hVal,biasHVal), axis=0)
            yVal= phi(W2@hVal)
            
            deltaY = (y-target)*phiPrime(yStar)
            deltaH = (W2.T@deltaY)[:nN,:] * phiPrime(hStar)
            dW1 = dW1*alpha - (deltaH@epoch.T)
            dW2 = dW2*alpha - (deltaY@h.T)
            W1 += dW1*eta
            W2 += dW2*eta
            
            #Evaluating how's the learning
            err = ((y-target)@(y-target).T)/2
            trainingError.append(err.squeeze().squeeze())    
            
            err = ((yVal-targetVal)@(yVal-targetVal).T)/2
            validationError.append(err.squeeze().squeeze())
            
    if method == 'sequential':
        for k in range(epochs):
            wellClassified = 0
            # Randomize the learning data and targets (not necessary when we do the batch algorithm)
            permutation = np.random.permutation(epoch.shape[1])
            epoch = epoch[:, permutation]
            target = target[:, permutation]
            for i in range(nA+nB):
                hStar = np.dot(W1, epoch[:,i])
                h = phi(hStar)
                yStar = np.dot(W2, h)
                y = phi(yStar)
                deltaY = np.multiply(np.add(-y, target[0,i]), phiPrime(yStar))
                deltaH = np.multiply(np.dot(np.transpose(W2), deltaY), phiPrime(hStar))
                W1 += eta*np.dot(deltaH, np.transpose(epoch[:,i]))
                W2 += eta*deltaY*np.transpose(h)
            #Evaluating how's the learning 
            for i in range(nA+nB):
                if y > 0:
                    if target[0, i] == 1:
                        wellClassified += 1
                else:
                    if target[0, i] == -1:
                        wellClassified += 1
            Accuracy.append(wellClassified/(nA+nB))
            Error.append(((y-target)@(y-target).T)/2)
    
            
    return  trainingError , validationError , W1,W2



def visualize(X,Y,Z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=1, aspect=5)
        ax.view_init(30, 35)
        plt.show()
        
def createDataForFuncAprox(vis = True):
    x = np.arange(-5,5,0.5)
    length = len(x)
    l2 = length*length
    y = np.arange(-5,5,0.5)
    xx, yy = np.meshgrid(x, y)
    z = np.exp(-0.1*xx*xx)*np.exp(-0.1*yy*yy)-0.5
    if vis:
        visualize(xx,yy,z) 
    targets = z.reshape((1,l2))
    inputs = np.concatenate((xx.reshape((1,l2)), yy.reshape((1,l2))),axis = 0)
    return inputs , targets ,length


def funcApproximate_3_2_3():
    eta= 0.001
    nN = 20
    epochs = 500
    method = 'batch'
    print('\n\nOriginal Function')
    patterns , targets ,l1= createDataForFuncAprox() 

   # permutation = np.random.permutation(patterns.shape[1])
    #patterns= patterns[:,permutation]
   # targets= targets[:,permutation]
    tEr,vEr,w1,w2 = trainFunctionApproximate(patterns,targets,eta,nN,method,epochs,perVal= 0.0) 
    newPatterns,a,b= createDataForFuncAprox(vis = False)
    x = patterns[0,:]
    y = patterns[1,:]
    x = x.reshape((l1,l1))
    y = y.reshape((l1,l1))
    output = funcAproxOutput(newPatterns,w1,w2)
    print('\n\n\n')
    print('Network output of the input when hidden nodes= ' , nN)
    output = output.reshape((l1,l1))
    visualize(x,y,output)
    visualizeTrained(tEr,epochs,'Training')
   # visualizeTrained(vEr,epochs,'Validation')


def visualizeTrained(Error,epochs,title):
    fig, (ax3) = plt.subplots(1, 1,figsize=(15,5),constrained_layout=True)
    ax3.plot(np.arange(epochs),Error, label='Error over iterations', color ='blue')
    fig.suptitle(title)
    ax3.legend()
    plt.show()


def funcAproxOutput(inp,W1,W2):
    def phi(x):
        return (2/(1+np.exp(-x))-1)
    inp = np.concatenate((inp,np.ones((1,inp.shape[1]))),axis= 0)
    hStar = W1@inp
    h = phi(hStar)
    biasH = np.ones([1,inp.shape[1]])
    h = np.concatenate((h,biasH), axis=0)
    yStar = W2@h
    y = phi(yStar)
    return y

