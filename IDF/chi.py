import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
 
n = 1000
proportion_true = 0.01
# A = np.random.choice([True, False], size=n, p=[proportion_true, 1-proportion_true])
 
# Create array with exact number of True and False values
num_true = int(n * proportion_true)
A = np.array([True] * num_true + [False] * (n - num_true))
# Shuffle to randomize the positions
np.random.shuffle(A)


future_proportion_true = 0.021
# B = np.random.choice([True, False], size=n, p=[future_proportion_true, 1-future_proportion_true])

future_num_true = int(n * future_proportion_true)
# Create array with exact number of True and False values
B = np.array([True] * future_num_true + [False] * (n - future_num_true))
# Shuffle to randomize the positions
np.random.shuffle(B)


# Chi-square test for independence
# Define the boolean vectors
 
# Count the number of ones and zeros in each vector
count_A_ones = np.sum(A)
count_A_zeros = len(A) - count_A_ones
count_B_ones = np.sum(B)
count_B_zeros = len(B) - count_B_ones
 
# Create a contingency table
contingency_table = np.array([[count_A_ones, count_A_zeros],
                              [count_B_ones, count_B_zeros]])
 
# Perform the chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
 
print(f"p-value: {p_value}")
 
 
#Two-Proportion Z-Test Approach
# Counts of ones
count = np.array([count_A_ones, count_B_ones])
# Number of observations
nobs = np.array([len(A), len(B)])
 
# Perform two-proportion z-test
stat, pval = proportions_ztest(count, nobs)
print(f"p-value: {pval}")
print(count)