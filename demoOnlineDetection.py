from __future__ import division
import numpy as np
from BOCD_Algorithms import *
from BOCD_modules import *
import matplotlib.pyplot as plt


"""
---------------------------------------------------------------------------------------------------------------------------
                                         Define the environment
---------------------------------------------------------------------------------------------------------------------------
"""

environment = np.array([0.9,0.1,0.8,0.2]) # Bernoulli distributions
Period = 300 # Length of each stationary period
environment = constructEnvironment(environment, Period) # Building the piece-wise stationary Bernoulli distributions


"""
----------------------------------------------------------------------------------------------------
                                Launch the change-point detection
----------------------------------------------------------------------------------------------------
"""

# Bayesian Online Change-point detector  without restart
Original = BOCD(environment) 
Modified = BOCDm(environment)

# Bayesian Online Change-point detector with restart
Original_restart = BOCD_restart(environment)
Modified_restart = BOCDm_restart(environment)
#print(environment)
#print(Original)


"""
----------------------------------------------------------------------------------------------------------------------------
                                       Plotting the results
----------------------------------------------------------------------------------------------------------------------------
"""

plt.plot(range(environment.size), Modified_restart.tolist(), color='red', marker='o', label = "BOCDm")
plt.hold(True)
plt.grid(True)
plt.plot(range(environment.size), Original_restart.tolist(), color='blue', marker='.', label = "BOCD")
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Change-point Estimation')
plt.title('with restart')
plt.show()
plt.hold(False)

plt.plot(range(environment.size), Modified.tolist(), color='red', marker='o', label = "BOCDm")
plt.hold(True)
plt.grid(True)
plt.plot(range(environment.size), Original.tolist(), color='blue', marker='.', label = "BOCD")
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Change-point Estimation')
plt.title('without restart')
plt.show()

