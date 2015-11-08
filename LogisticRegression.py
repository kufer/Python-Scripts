# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:55:43 2015

@author: kushal
"""

from sklearn import linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("Nomis_data.csv", delimiter = ',', usecols = (0, 4, 8, 9, 7), skiprows = 1)
#import the columns tier, amount, apr, prime, and outcome from nomis_data.csv


independent_vars = data[ : , 0:4]
#2D array of independent variables tier, amount, apr, prime

accept = (data[ : , 4] > 0)
#Create an array containing outcomes = 1

logit = lm.LogisticRegression()
#Create an object of class LogisticRegression from sklearn

logit.set_params(C = 20, intercept_scaling = 20)
#Reduce "regularization" to get vanilla logistic regression

logit.fit(independent_vars, accept)
#Use the fit member function to fit a logistic regression

accept_pred = logit.predict_proba(independent_vars)
#Use the predict_proba member function to calculate the predicted acceptance proba

#Print the coeffecients of regressed dependent variables, and model RMSE 
print("Regressed Coeffecients of [Tier, Amount, APR, Prime]")
np.set_printoptions(precision = 4, suppress = True)
print(np.array(*logit.coef_))
print("RMSE of logistic prediction of probability is:", '%.3f' %np.std(accept - accept_pred[ : , 1]))

fraction_accept = [0, 0, 0, 0, 0]
bins = [0.0, 0.2, 0.4, 0.6, 0.8]
#list to hold acceptance fractions for each bin

for i in range(0, 5):
    #loop through each bin
    l = 0
    m = 0
    #intitialize counter variables for calculating fraction of acceptance
    for j in range(0, data.shape[0]):
        #loop through each row of data
        if(bins[i] <= accept_pred[j, 1] < bins[i] + 0.2): 
            #check if the predicted acceptance lies in the bin range
            l = l + 1    
            #increment a counter variable for each such occurence
            if(accept[j] == 1):
                #further, check if the given row has an outcome/acceptance of 1
                m = m + 1
                #increment a counter representing accept = 1 and accept_pred in the current bin
    fraction_accept[i] = m / l
    #calculate the fraction for the 'i'th bin

#plot a bar chart of the fractions vs bins using matplotlib
plt.figure(figsize = (8, 5))
plt.xlabel('Probability of acceptance from model')
plt.ylabel('Fraction of acceptance from data')
plt.bar(bins, fraction_accept, 0.2)
plt.plot([0, 1], [0, 1])
plt.grid(b = True, which = 'major', linestyle = '--')
plt.grid(b = True, which = 'minor', linestyle = '--')
plt.show()
        