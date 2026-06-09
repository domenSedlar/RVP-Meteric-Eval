import itertools
import numpy as np

def eval_matrix_full(A):
    (n,m) = A.shape

    results = []

    for p in itertools.permutations(range(n)):
        for r in itertools.permutations(range(m)):
            a = A[np.ix_(p, r)]