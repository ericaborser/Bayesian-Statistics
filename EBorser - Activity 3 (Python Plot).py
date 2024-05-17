"""
Created on Sat Feb 17 16:56:44 2024

@author: Erica Borser
"""

import numpy as np
import matplotlib.pyplot as plt

prior_prob = np.array([[0.33,0.3],[0.2,0.17]])

plt.imshow(prior_prob, cmap='gray')
plt.colorbar()

for i in range (2):
    for j in range (2):
        plt.annotate(prior_prob[i,j], [j,i], color="red", fontsize=20, fontweight='bold', ha='center', va='center')
        
plt.title('Prior Probabilities', fontsize=20)
