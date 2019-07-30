# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:49:31 2019

@author: Leon
"""

#imports
import numpy as np
import numpy.matlib
import math
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

initialTrain = np.genfromtxt ('letters.csv', delimiter=",")
trainlabels = np.genfromtxt ('letterslabels.csv', delimiter=",")


totalDeadUnits = np.zeros((1,10))

for z in range(10):

    train = initialTrain
    #set the seed for the numpy random functions (used to create same conditions)
    np.random.seed(z)

    [n,m]  = np.shape(train)                    # number of pixels and number of training data
    pix = int(math.sqrt(n))
    image = np.reshape(train[:,10],(pix,pix),order="F")

    [n,m]  = np.shape(train)                    # number of pixels and number of training data
    #number of pixels is 7744 (88*88)
    #number of training data examples is 7000

    eta    = 0.05                              # learning rate
    winit  = 1                                 # parameter controlling magnitude of initial conditions
    alpha = 0.999

    tmax   = 20000                            #number of training instances
    digits = 20                               #number of digits

    wMatrix = winit * np.random.rand(digits,n)        # Weight matrix (rows = output neurons, cols = input neurons)
    normW = np.sqrt(np.diag(wMatrix.dot(wMatrix.T)))
    normW = normW.reshape(digits,-1)  # reshape normW into a numpy 2d array
    wMatrix = wMatrix / normW                           # normalise using numpy broadcasting -  http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html

    #normalise the dataset
    norm = np.atleast_1d(np.linalg.norm(train, ord =2, axis =0))
    norm[norm==0] = 1
    train = train / np.expand_dims(norm, axis = 0)
    [n,m]  = np.shape(train)

    counter = np.zeros((1,digits))              # counter for the winner neurons
    wCount = np.ones((1,tmax+1)) * 0.25         # running avg of the weight change over time

    rate=1
    for t in range(1,tmax):
        i = math.ceil(m * np.random.rand())-1  # get a randomly generated index in the input range
        x = train[:,i]                          # pick a training instance using the random index

        h = wMatrix.dot(x)/digits               # get output firing
        h = h.reshape(h.shape[0],-1)            # reshape h into a numpy 2d array

        output = np.max(h)                      # get the max in the output firing vector
        k = np.argmax(h)                        # get the index of the firing neuron

        counter[0,k] += 1                       # increment counter for winner neuron

        dw = eta * (x.T - wMatrix[k,:])         # calculate the change in weights for the k-th output neuron
                                                # get closer to the input (x - W)

        wCount[0,t] = wCount[0,t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # % weight change over time (running avg)

        wMatrix[k,:] = wMatrix[k,:] + dw                # weights for k-th output are updated

        ### leaky learning
        for s in range(0, digits):                      #loop over the number of neurons
            if (s != k):
                low_eta = eta / 1000                   #set a learning rate that is much lower than that of the winner neuron
                dw = low_eta * (x.T - wMatrix[s,:])     #get the change in weight for neuron index s
                wMatrix[s,:] += dw                      #update the weights of the s indexed neuron


        #adding noise to weights
        mu, sigma = 0, 0.00005                             #set a noise value threshold
        noise = np.random.normal(mu, sigma, wMatrix.shape) #generate a noise matrix the same size as the weight matrix
        wMatrix[k,:] += noise[k,:]                         #apply the noise matrix to the weights matrix

    #calculate correlation matrix
    correlation_matrix = np.corrcoef(wMatrix)

    [i,j]  = np.shape(correlation_matrix)
    counter = 0
    deadUnits = np.zeros((1,digits))

    sums = wMatrix.sum(axis=1)

    for y in range(j):
        if (sums[y] > 40):
                deadUnits[0,y] += 1
                counter +=1

    #calculates the number of dead units by recursing through the correlation matrix and
    #comparing the correlation between prototypes. If a prototype is has a threshold higher than
    #0.8 when compared to a previous non-dead unit is classified as a dead unit.


    for y in range(j):
        if (deadUnits[0,y] == 0) :
            for x in range(i):
                if (deadUnits[0,x] == 0):
                    if (correlation_matrix[x,y] > 0.8 and correlation_matrix[x,y] != 1):
                        deadUnits[0,x] += 1
                        counter +=1

    totalDeadUnits[0,z] = counter
print("noise and leaky learning:")
print(totalDeadUnits)
