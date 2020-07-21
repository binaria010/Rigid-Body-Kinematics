#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:49:55 2020

@author: Juliana
"""

import numpy as np
import matplotlib.pyplot as plt

def euler_integrator(function, x0, t_span, step, args= None):
    """
    this function solves an ode using forward euler method: X'=F(X,t)
    then X' = (X(t+1) - X(t))/(Delta t) therefore X(t+1)= X(t) + F(X(t),t)*Delta t
    Parameters
    ----------
    function : function. Right hand side of the equation 
    x0 : initial condition. 
    t_span : Tuple -> (t_0, t_f)
    step: float -> is the Delta t value in the forward euler approximation
    args : 

    Returns
    -------
    ndarray of size len(x0)xlen(t)-> the values of the solution for each time t.

    Example:

    x0 = np.array([1])
    t_span = (0, np.pi)

    step = 0.1

    def function(x,t) :
        return -np.sin(t)

    X, t = euler_integrator(function, x0, t_span, step)

    plt.figure(figsize = (10,6))
    plt.plot(t, X.T)

    """
    x = x0
    t0, t_f = t_span
    t = np.arange(t0, t_f, step)
    nt = t.shape[0]
    X = np.zeros((x0.shape[0], nt))
    X[:,0] = x


    
    for i in range(nt-1):
        F = function(t[i],x)
        x = x + F* step
        X[:,i+1] = x
    return X,t



def ODE4_order(function, x0, t_span, step, args = None):
    """
    This function integrates an ODE using the 4 order Runge-Kutta method.
    
    Parameters
    ----------
    function : function. Right hand side of the equation 
    x0 : initial condition. 
    t_span : Tuple -> (t_0, t_f)
    step: float -> is the Delta t value in the forward euler approximation
    args : 

    Returns
    -------
    X : ndarray of size len(x0)xlen(t)-> the values of the solution for each time t.   
    """
    
    x = x0
    t0, t_f = t_span
    t = np.arange(t0, t_f, step)
    nt = t.shape[0]
    X = np.zeros((x0.shape[0],nt))
    X[:,0] = x
    
    for i in range(nt-1):
        k1 = function(t[i],x)
        k2 = function(t[i] + 0.5*step, x + 0.5*step*k1)
        k3 = function(t[i] + 0.5*step, x + 0.5*step*k2)
        k4 = function(t[i] + step, x + step*k3)
        x = x + step*(k1 +2*k2 + 2*k3 + k4)/6
        X[:,i+1] = x
        
    return X,t

