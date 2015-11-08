# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:51:23 2015

@author: kushal
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model as lm

def robust_linear_reg(x, y):
    """
        Robust least squares.
    
        minimize  sum( h( y - bx ))
        
        where h(u) = u^2           if |u| <= 1.0
                   = 2*(|u| - 1.0) if |u| > 1.0.
        Input:
            y: 1D dependent variables
            x: 1D independent variables
    """
    # 1. Define objective function
    
    # function to determine the correct mathematical function
    def h(u):
        
        if(abs(u) <= 1.0):
            u = u**2
        else:
            u = 2 * (abs(u) - 1.0)
        return u     
    
    # vectorize the above function so that we can compare element wise
    h = np.vectorize(h)  
    
    # define objectives using lambda function
    obj = lambda b: sum(h(y - b[0] - b[1] * x))        
    
    # 2. Set initial guess 
    b = [0, 0]
    
    # 3. solve the optimization problem
    res = minimize(obj, b, method = 'SLSQP')
    
    # Return coefficients
    return res.x
    
# Read data for test
data = np.loadtxt("robust_reg_data.csv", delimiter = ',')
numObs = len(data)
x = data[ : , 0:1]
y = data[ : , 1:2]

# Run ordinary (least-square) linear regression
linear = lm.LinearRegression()
linear.fit(x, y)
line_y = linear.predict(x)

# Run robust linear regression
robust = robust_linear_reg(x, y)
line_y_robust = robust[0] + robust[1] * x

print("The linear regression slopes are - ordinary:", "%.3f" %linear.coef_, "and robust:", "%.3f" %robust[1])
print("The linear regression intercepts are - ordinary:", "%.3f" %linear.intercept_, "and robust:", "%.3f" %robust[0])

# Create a plot to visualize the difference
plt.scatter(x, y, marker = "o")
plt.plot(x, line_y, '-k', label = 'Linear Fit')
plt.plot(x, line_y_robust, '-b', label = 'Robust Fit')
plt.legend(loc = 'upper center')
plt.xlim(min(x), max(x))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

