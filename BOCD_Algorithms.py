from __future__ import division
import numpy as np
from BOCD_modules import *

import matplotlib.pyplot as plt


"""
--------------------------------------------------------------------------------------------------------------------------------------

Bayesian Online Change-point Detection original version

 Inputs:  
  -- environment: numpy array of piece-wise Bernoulli distributions

 Outputs:
  -- ChangePointEstimation: numpy array of change-point estimations

--------------------------------------------------------------------------------------------------------------------------------------
"""

def BOCD(environment):
    #--------------- Initialization ---------------------
    Horizon = environment.size
    gamma = 1/Horizon # Switching Rate 
    alphas = np.array([1])
    betas = np.array([1])
    ForecasterDistribution = np.array([1])
    ChangePointEstimation = np.array([])
    #-----------------------------------------------------
    #Interation with the environment ...
    print('Launching BOCD ...')
    for t in range(Horizon):
        EstimatedBestExpert = np.argmax(ForecasterDistribution) #Change-point estimation
        ChangePointEstimation = np.append(ChangePointEstimation,EstimatedBestExpert+1)
        reward = int ((np.random.uniform() < environment[t]) == True) #Get the observation from the environment
        ForecasterDistribution = updateForecasterDistribution(ForecasterDistribution, alphas, betas, reward, gamma)
        (alphas, betas) = updateLaplacePrediction(alphas, betas, reward) #Update the laplace predictor
    return ChangePointEstimation	


"""
--------------------------------------------------------------------------------------------------------------------------------------

Bayesian Online Change-point Detection with original prior and restart

 Inputs:  
  -- environment: numpy array of piece-wise Bernoulli distributions

 Outputs:
  -- ChangePointEstimation: numpy array of change-point estimations

--------------------------------------------------------------------------------------------------------------------------------------
"""


def BOCD_restart(environment):
    #--------------- Initialization ---------------------
    Horizon = environment.size
    gamma = 1/Horizon # Switching Rate 
    alphas = np.array([1])
    betas = np.array([1])
    ForecasterDistribution = np.array([1])
    ChangePointEstimation = np.array([])
    Restart = 1  # Position of last restart
    #-----------------------------------------------------
    #Interation with the environment ...
    print('Launching BOCD with restart ... ')
    for t in range(Horizon):
        EstimatedBestExpert = np.argmax(ForecasterDistribution)
        # Restart precedure
        if not(EstimatedBestExpert == 0):
            # Reinitialization
            alphas = np.array([1]) 
            betas = np.array([1])
            ForecasterDistribution = np.array([1])
            Restart = t+1
        ChangePointEstimation = np.append(ChangePointEstimation,Restart+1)#Change-point estimation
        reward = int ((np.random.uniform() < environment[t]) == True) #Get the observation from the environment
        ForecasterDistribution = updateForecasterDistribution(ForecasterDistribution, alphas, betas, reward, gamma)
        (alphas, betas) = updateLaplacePrediction(alphas, betas, reward) #Update the laplace predictor
    return ChangePointEstimation	



"""
--------------------------------------------------------------------------------------------------------------------------------------

Bayesian Online Change-point Detection modified without restart and simple prior

 Inputs:  
  -- environment: numpy array of piece-wise Bernoulli distributions

 Outputs:
  -- ChangePointEstimation: numpy array of change-point estimations

--------------------------------------------------------------------------------------------------------------------------------------
"""

def BOCDm(environment):
    #--------------- Initialization ---------------------
    Horizon = environment.size
    gamma = 1/Horizon # Switching Rate 
    alphas = np.array([1])
    betas = np.array([1])
    ForecasterDistribution = np.array([1])
    PseudoDist = np.array([1])
    ChangePointEstimation = np.array([])
    like1 = 1
    #-----------------------------------------------------
    #Interation with the environment ...
    print('Launching BOCD modified ... ')
    for t in range(Horizon):
    	EstimatedBestExpert = np.argmax(ForecasterDistribution)
    	ChangePointEstimation = np.append(ChangePointEstimation,EstimatedBestExpert+1) #Change-point estimation
    	reward = int ((np.random.uniform() < environment[t]) == True) #Get the observation from the environment
    	(ForecasterDistribution, PseudoDist, like1) = updateForecasterDistribution_m(ForecasterDistribution,PseudoDist, alphas, betas, reward, gamma, like1)
    	(alphas, betas) = updateLaplacePrediction(alphas, betas, reward) #Update the laplace predictor
    return ChangePointEstimation	



"""
--------------------------------------------------------------------------------------------------------------------------------------

Bayesian Online Change-point Detection modified without restart and simple prior

 Inputs:  
  -- environment: numpy array of piece-wise Bernoulli distributions

 Outputs:
  -- ChangePointEstimation: numpy array of change-point estimations

--------------------------------------------------------------------------------------------------------------------------------------
"""


def BOCDm_restart(environment):
    #--------------- Initialization ---------------------
    Horizon = environment.size
    gamma = 1/Horizon # Switching Rate 
    alphas = np.array([1])
    betas = np.array([1])
    ForecasterDistribution = np.array([1])
    PseudoDist = np.array([1])
    ChangePointEstimation = np.array([])
    like1 = 1
    Restart = 1  # Position of last restart
    #------------------------------------------------------
    #Interation with the environment ...
    print('Launching BOCD modified with restart ...')
    for t in range(Horizon):
        EstimatedBestExpert = np.argmax(ForecasterDistribution)
        # Restart precedure
        if not(EstimatedBestExpert == 0):
            # Reinitialization
            alphas = np.array([1])
            betas = np.array([1])
            ForecasterDistribution = np.array([1])
            Restart = t+1
            like1 = 1
        ChangePointEstimation = np.append(ChangePointEstimation,Restart+1) #Change-point estimation
        reward = int ((np.random.uniform() < environment[t]) == True) #Get the observation from the environment
        (ForecasterDistribution, PseudoDist, like1) = updateForecasterDistribution_m(ForecasterDistribution,PseudoDist, alphas, betas, reward, gamma, like1)
        (alphas, betas) = updateLaplacePrediction(alphas, betas, reward) #Update the laplace predictor
    return ChangePointEstimation	


