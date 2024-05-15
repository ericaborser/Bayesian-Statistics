# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:54:30 2024

@author: Erica
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# Generate some synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_slope = 2
true_intercept = 5
true_sigma = 2
Y = true_slope * X + true_intercept + np.random.normal(scale=true_sigma, size=len(X))

# Plot the data
plt.scatter(X, Y, label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Synthetic Data')
plt.legend()
plt.show()

# Define the Bayesian regression model
with pm.Model() as model:
    # Priors
    slope = pm.Normal('slope', mu=0, sd=10)
    intercept = pm.Normal('intercept', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=10)

    # Expected value of outcome
    mu = slope * X + intercept

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

# Perform Bayesian inference
with model:
    trace = pm.sample(1000, cores=1)  # Change cores to the number of CPU cores you have

# Plot the posterior distributions
pm.traceplot(trace)
plt.show()
