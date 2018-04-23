"""Helper functions that are common for both Gibbs samplers for LDA."""
from collections import defaultdict
import numpy as np


def randinit_z(data, K):
    """Randomly initialize Z from a uniform distribution. Z is represented as 
    a dictionary, where Z[d][w] is a numpy array where the i-th element contains 
    the topic corresponding to the i-th instance of the word w in document d."""
    Z = defaultdict(lambda: defaultdict(np.array))
    for d, w, train_count, _ in data:
        Z[d-1][w-1] = np.random.choice(K, train_count)
    return Z


def calc_counts(Z, K, W):
    """Calculates the counts A, B & M from topic indicators Z."""
    A = np.zeros((len(Z), K), dtype=np.int)  # element (d,k) holds the number of words in document d assigned to topic k
    B = np.zeros((K, W), dtype=np.int)  # element (k, w) holds the number of times word w is assigned to topic k across all documents
    M = np.zeros(K, dtype=np.int)  # element k holds the total number of words assigned to topic k across all documents

    for k in range(K):
        for d in Z:
            for w in Z[d]:
                total = sum(Z[d][w] == k)
                A[d, k] += total
                B[k, w] += total
                M[k] += total
    
    return A, B, M
    
