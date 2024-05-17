# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:59:23 2024

@author: Erica Borser
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior parameters for the uniform distribution
a_prior = 1 # Alpha parameter for the uniform distribution
b_prior = 1 # Beta parameter for the uniform distribution

# Data: number of positive test results and total individuals tested
positive_tests = 40
total_tests = 50

# Likelihood function: Binomial distribution
likelihood = lambda theta: theta**positive_tests * (1-
theta)**(total_tests - positive_tests)

# Posterior parameters calculation
a_posterior = a_prior + positive_tests
b_posterior = b_prior + total_tests - positive_tests

# Define the posterior distribution (Beta distribution)
posterior = beta(a_posterior, b_posterior)

# Plotting
theta_values = np.linspace(0, 1, 1000)
prior_pdf = beta(a_prior, b_prior).pdf(theta_values)
likelihood_values = likelihood(theta_values)
posterior_pdf = posterior.pdf(theta_values)

plt.figure(figsize=(10, 6))
plt.plot(theta_values, prior_pdf, label='Prior', 
linestyle='--')
plt.plot(theta_values, likelihood_values, 
label='Likelihood', linestyle='-.')
plt.plot(theta_values, posterior_pdf, label='Posterior')
plt.xlabel('θ (Probability of positive test)')
plt.ylabel('Density')
plt.title('Bayesian Inference: Test Result Accuracy')
plt.legend()
plt.show()
