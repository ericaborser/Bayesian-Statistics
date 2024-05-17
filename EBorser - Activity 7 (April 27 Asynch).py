"""
Created on Mon Apr 29 16:40:38 2024

@author: Erica Borser
#CAS-05-601A
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid", palette="muted")

def generate_linear_data(
        start, stop, num_points, intercept, slope, noise_mean, noise_variance, random_seed
):
    """
    Generate a random dataset using a linear process with noise.

    Parameters
    ----------
    start : float
        Start value for the predictor variable
    stop : float
        Stop value for the predictor variable
    num_points : int
        Number of data points to generate
    intercept : float
        Intercept of the linear relationship
    slope : float
        Slope of the linear relationship
    noise_mean : float
        Mean of the noise
    noise_variance : float
        Variance of the noise
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the predictor and response variables.

    """
    np.random.seed(random_seed)
    x = np.linspace(start, stop, num=num_points)
    y = intercept + slope * x + np.random.normal(noise_mean, np.sqrt(noise_variance), size=num_points)
    df = pd.DataFrame({"x": x, "y": y})
    return df

def plot_data(df):
    """
    Plot the generated data using seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the predictor and response variables.

    """
    sns.lmplot(x="x", y="y", data=df, height=10)
    plt.xlim(0.0, 1.0)
    plt.show()

if __name__ == "__main__":
    intercept = 1.0
    slope = 2.0
    start = 0
    stop = 1
    num_points = 100
    noise_mean = 0.0
    noise_variance = 0.5
    random_seed = 42
    
    df = generate_linear_data(start, stop, num_points, intercept, slope, noise_mean, noise_variance, random_seed)
    plot_data(df)
