# -*- coding: utf-8 -*-
"""
knn_insample.py
Created on Sun Oct 26 12:51:35 2015

@author: Kushal Fernandes, Dongwook Shin, Yash Kanoria

User input (in the main function):
    - 'facebook_data.csv' 
       Data samples in rows, outcome variable (friend or not) in first col
    - 'facebook_data_test.csv'  
       Data samples in rows (mutual friends, distance)
    -  k: the number of nearest neighbors desired

Output:
    - 2(a) 
        - [x_sample, y_sample, y_pred]: in-sample data read from the file 
                                        and computed prediciton
        - rmse: in-sample rmse of prediction
    - 2(b)
        - plot of rmse vs k for k in range 1 to 40
    - 2(c)
        - [normal_x, y_sample, y_pred]: in-sample kNN using 
                                        normalized predictors
        - rmse_normalized: in-sample rmse of normalized prediction
        - plot of normalized rmse vs k for k in range 1 to 40
    - 2(d)
        - [x_train, y_train]: train data read from the file
        - [x_test, y_test]: test data matrix, x_test, and
                            corresponding predicted values, y_test
"""

import numpy as np
import scipy.spatial.distance as ssd
import time
import matplotlib.pyplot as plt

def knn_insample_predict(x_sample, y_sample, k):
    """ function for out-of-sample prediction using k-NN algorithm 
        Inputs:            
            x_sample: 2D array containing samples in rows
            y_sample: 1D array containing outcomes for samples
            k: Number of neighbors to use
        
        Output:
            List containing in-sample predictions and rmse
    """
    # initialize lists to store predicted value
    y_pred = []
    
    # for each row in sample data, calculate distance wrt to other rows
    for i, di in enumerate(x_sample):
        distances = []  # initialize list to store distance
        for j, dj in enumerate(x_sample):
            # if current row is same as in-sample row, skip to next row
            if j == i:
                continue
            else:
                distances.append((ssd.euclidean(di,dj), y_sample[j]))
        # k-neighbors
        sorted_distances = sorted(distances)[:k]

        # predict the outcome for the instance
        y_pred.append(np.mean(sorted_distances, axis = 0)[1])
    
    #compute the rmse for all row predictions
    rmse = np.sqrt(((y_pred - y_sample) ** 2).mean())
    
    # return predicted outcome and rmse
    return y_pred, rmse 

""" The main code starts here """
# string of dahes to separate each answer block
dashes = '-------------------------'

# initialize runtime
start = time.clock()

# read sample data and test data from .csv files
sample_data = np.loadtxt('facebook_data.csv', delimiter = ',', dtype = "float")
test_data = np.loadtxt('facebook_data_test.csv', delimiter = ',', dtype = "float")

# translate sample_data to x_sample, x_test and y_sample array
y_sample = sample_data[:, 0]
x_sample = sample_data[:, 1:]
x_test = test_data[:, :]

print('\n' + dashes + 'begin 2(a)' + dashes)

# set k, the number of nearest neighbors desired
k = 3

# run k-NN algorithm to predict y_pred in-sample for sample_data
y_pred, rmse = knn_insample_predict(x_sample, y_sample, k)

#print out results and rmse
print("\nk-NN in sample prediction results for sample data with k = %d" % k)
print("\nx_sample \t | y_sample \t | y_predicted")
for i, di in enumerate(x_sample):
    a = ','.join('{:5.0f}'.format(k) for ik, k in enumerate(di))
    strRow = a + "\t | " + '{:5.0f}'.format(y_sample[i]) + "\t | " + '{:5.4f}'.format(y_pred[i])
    print(strRow)
print("\nRMSE of prediction is = %5.4f" %rmse)

# Find runtime
run_time = time.clock() - start
print("Runtime:", '%.4f' % run_time, "seconds")

print('\n' + dashes + 'end 2(a)' + dashes)
 
#2(b)
print('\n' + dashes + 'begin 2(b)' + dashes)
rmse_different_k = [] #list to hold the rmse for different k values 
k_values = [i for i in range(1, 40)] # populate a list with desired k values

# for each k value, run in-sample kNN and record the rmse
for k in k_values:
    _, rmse_k = knn_insample_predict(x_sample, y_sample, k)
    # _, rmse_k means that the first return value is ignored
    rmse_different_k.append(rmse_k)

# test_k holds the k value corresponding to the lowest rmse
test_k = rmse_different_k.index(min(rmse_different_k)) + 1 
# +1 because k index starts from 1 but list index starts from 0

#plot rmse vs k
plt.figure(1)
plt.plot(k_values, rmse_different_k, '-k')
plt.title('2(b): RMSE vs k for k in range 1 to 40')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.show()
print('\n Graphical Plot of RMSE vs k')
print('\n' + dashes + 'end 2(b)' + dashes)

#2(c)
print('\n' + dashes + 'begin 2(c)' + dashes)
k_norm = 3 #k value for the normalized prediction
x_sample_normalized = [] #list to hold normalized predictors

# initialize runtime
start = time.clock()

predictors_mean = np.mean(x_sample, axis = 0) #mean of the predictors
predictors_range = np.ptp(x_sample, axis = 0) #range of the predictors

#normalize each row of predictors, subtracting the mean and dividing by range
for i, di in enumerate(x_sample):
    x_sample_normalized.append((di - predictors_mean)/predictors_range)

#compute the in-sample prediction and rmse using normalized predictors
y_pred_norm, rmse_normalized = knn_insample_predict(x_sample_normalized, y_sample, k_norm)

#print out normalized results and rmse
print("\nk-NN in sample prediction results for normalized data with k = %d" % k_norm)
print("\nnormal_x \t\t | y_sample \t | y_predicted")
for i, di in enumerate(x_sample_normalized):
    a = ','.join('{:5.4f}'.format(k) for ik, k in enumerate(di))
    strRow = a + "\t | " + '{:5.0f}'.format(y_sample[i]) + "\t | " + '{:5.4f}'.format(y_pred_norm[i])
    print(strRow)
print("\nRMSE of normalized prediction is = %5.4f" %rmse_normalized)

# Find runtime
run_time = time.clock() - start
print("Runtime:", '%.4f' % run_time, "seconds")

#Compute normalized prediction rmse for different values of k, same as in 2(b)
rmse_different_k_norm = []
k_values = [i for i in range(1, 40)]

for k in k_values:
    _, rmse_k_norm = knn_insample_predict(x_sample_normalized, y_sample, k)
    rmse_different_k_norm.append(rmse_k_norm)

#plot normalized rmse vs k
plt.figure(2)
plt.plot(k_values, rmse_different_k_norm, '-k')
plt.title('2(c): Normalized predictors RMSE vs k for k in range 1 to 40')
plt.xlabel('k')
plt.ylabel('Normalized RMSE')
plt.show()

print('\nGraphical Plot of Normalized RMSE vs k\n')

#compare the normalized rmse vs the vanilla in-sample rmse and print the outcome
if rmse_normalized > rmse:
    print("The RMSE using normalized predictors is", '%5.4f' %rmse_normalized, "which is worse than the un-normalized RMSE of", '%5.4f' %rmse)
elif rmse_normalized == rmse:
    print("The RMSE using normalized predictors is", '%5.4f' %rmse_normalized, "which is equal to the un-normalized RMSE of", '%5.4f' %rmse)
else:
    print("The RMSE using normalized predictors is", '%5.4f' %rmse_normalized, "which is better than the un-normalized RMSE of", '%5.4f' %rmse)

print('\n' + dashes + 'end 2(c)' + dashes) 

#2(d)   
print('\n' + dashes + 'begin 2(d)' + dashes)
#out-of-sample kNN using training and test sets

def knn_predict(x_train, y_train, x_test, k):
    """ function for out-of-sample prediction using k-NN algorithm 
        Inputs:            
            x_train: 2D array containing training samples in rows
            y_train: 1D array containing outcomes for training samples
            x_test:  2D array containing test samples in rows
            k:       Number of neighbors to use
        
        Output:
            List containing predictions for test samples
    """
    # initialize list to store predicted class
    y_test = []
    
    # for each instance in data testing,
    # calculate distance in respect to data training
    for i, di in enumerate(x_test):
        distances = []  # initialize list to store distance
        for j, dj in enumerate(x_train):
            # calculate distances
            distances.append((ssd.euclidean(di,dj), y_train[j]))
        # k-neighbors
        sorted_distances = sorted(distances)[:k]

        # predict the outcome for the instance
        y_test.append(np.mean(sorted_distances, axis = 0)[1])
       
    # return predicted outcome
    return y_test 

# initialize runtime
start = time.clock()    

#compute predicted test values based on training data
y_test = knn_predict(x_sample, y_sample, x_test, test_k)
#test_k computed as k which minimizes rmse in step 2(b)

# print out results
print("\nk-NN train data:")
print("x_train \t | y_train")
for i, di in enumerate(x_sample):
    a = ','.join('{:5.0f}'.format(k) for ik, k in enumerate(di))
    strTrainRow = a + "\t | " + '{:5.0f}'.format(y_sample[i])
    print(strTrainRow)

print("\nk-NN prediction results for test data with k = %d" % test_k)
print("x_test \t | y_test")
for i, di in enumerate(x_test):
    a = ','.join('{:5.0f}'.format(k) for ik, k in enumerate(di))
    strTestRow = a + "\t | " + '{:5.0f}'.format(round(y_test[i]), decimals = 0)
    print(strTestRow)
 
# # Find runtime
run_time = time.clock() - start
print("\nRuntime:", '%.4f' % run_time, "seconds")

print('\n' + dashes + 'end 2(d)' + dashes) 