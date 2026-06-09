import itertools
import numpy as np
from RVP_Metrics.metrics import full_eval as score
import random
import pandas as pd
import os
import argparse

def eval_matrix(A, permutations1, permutations2):
    results = []

    for p1 in permutations1:
        for p2 in permutations2:
            a = A[np.ix_(p1, p2)]
            r = score(a)
            r['p1'] = tuple(p1)
            r['p2'] = tuple(p2)
            results.append(r)

    return results

def full_eval(A):
    (m,n) = A.shape
    p1 = list(itertools.permutations(range(m)))
    p2 = list(itertools.permutations(range(n)))
    return eval_matrix(A, p1, p2)

def random_eval(A):
    (m,n) = A.shape

    p1 = [random.shuffle(range(m)) for _ in range(100)]
    p2 = [random.shuffle(range(n)) for _ in range(100)]

    return eval_matrix(A)

def get_results(pth):
    A = np.loadtxt(pth, delimiter='\t')

    output_file = pth.split('\\')[-1].split("/")[-1].split('.')[0] + '.csv'

    (m,n) = A.shape

    if max(m,n) < 10:
        r = full_eval(A)
    else:
        r = random_eval(A)
    df = pd.DataFrame(r)
    df.to_csv(os.path.join("./Results/", output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that checks and scores permutations of a provided matrix")
    parser.add_argument("path", help="The path to the input .tsv file containing the matrix")

    args = parser.parse_args()

    get_results(args.path)
    print("saved the data")