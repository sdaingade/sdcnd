import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    denominator = np.sum(np.exp(L))
    return [np.exp(i)/denominator for i in L]

