from __future__ import division
import numpy as np

import sys


#-------------------------------------------------------------------------------------------------------------------
#                       Building the piece-wise stationary Bernoulli distributions
#-------------------------------------------------------------------------------------------------------------------
def constructEnvironment(environment, Period):
    vect = np.array([])
    for periode in range(environment.size):
        vect = np.append(vect, environment[periode]*np.ones((Period)))
    return vect


#-------------------------------------------------------------------------------------------------------------------
#                       Updating the forecaster distribution using the message passing algorithm
#-------------------------------------------------------------------------------------------------------------------

def updateForecasterDistribution(ForecasterDistribution, alphas, betas, reward, gamma):
    if reward == 1:
        likelihood = np.divide(alphas, alphas + betas)
    else:
        likelihood = np.divide(betas, alphas + betas)
    ForecasterDistribution0 = gamma*np.dot(likelihood, np.transpose(ForecasterDistribution)) # Creating new Forecaster 
    ForecasterDistribution = (1-gamma)*likelihood*ForecasterDistribution # update the previous forecasters 
    ForecasterDistribution = np.append(ForecasterDistribution,ForecasterDistribution0) # Including the new forecaseter into the previons ones
    ForecasterDistribution = ForecasterDistribution/np.sum(ForecasterDistribution) # Normalization for numerical purposes
    return ForecasterDistribution


#-------------------------------------------------------------------------------------------------------------------
#                       Updating the forecaster distribution using the message passing algorithm with a modified prior (q)
#-------------------------------------------------------------------------------------------------------------------

def updateForecasterDistribution_m(ForecasterDistribution, PseudoDist, alphas, betas, reward, gamma, like1):
    if reward == 1:
        likelihood = np.divide(alphas, alphas + betas)
    else:
        likelihood = np.divide(betas, alphas + betas)
    Pseudo_w0 = gamma*like1*np.sum(PseudoDist) # Using the simple prior
    PseudoDist = like1*PseudoDist
    ForecasterDistribution0 = Pseudo_w0 # Creating new Forecaster
    ForecasterDistribution = (1-gamma)*likelihood*ForecasterDistribution # update the previous forecasters
    ForecasterDistribution = np.append(ForecasterDistribution,ForecasterDistribution0) # Including the new forecaseter into the previons ones
    ForecasterDistribution = ForecasterDistribution/np.sum(ForecasterDistribution) # Normalization for numerical purposes
    PseudoDist = np.append(PseudoDist,Pseudo_w0)
    PseudoDist = PseudoDist/np.sum(PseudoDist) # Normalization for numerical purposes
    return ForecasterDistribution, PseudoDist, like1


#-------------------------------------------------------------------------------------------------------------------
#                       Updating the laplace prediction for each forecasters
#-------------------------------------------------------------------------------------------------------------------

def updateLaplacePrediction(alphas, betas, x):
    alphas[:] += x 
    betas[:] += (1-x)
    alphas = np.append(alphas,1) # Creating new Forecaster
    betas = np.append(betas,1) # Creating new Forecaster
    return alphas, betas


