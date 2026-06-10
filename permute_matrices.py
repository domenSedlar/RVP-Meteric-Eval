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
    permutations1 = list(itertools.permutations(range(m)))
    permutations2 = list(itertools.permutations(range(n)))
    results = []
    total = len(permutations1) * len(permutations2)
    progress_interval = 100000
    current = 0
    for p1 in permutations1:
        for p2 in permutations2:

            current += 1

            if current % progress_interval == 0:
                print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

            a = A[np.ix_(p1, p2)]
            r = score(a)
            r['p1'] = tuple(p1)
            r['p2'] = tuple(p2)
            results.append(r)

    return results

def random_eval(A):
    (m,n) = A.shape
    k = 40000
    permutations = [(random.sample(range(m), m), random.sample(range(n), n)) for _ in range(k)]    
    
    results = []

    for i, (p1, p2) in enumerate(permutations):
        
        if i % 100000 == 0:
            print(f"Progress: {i}/{k} ({i/k*100:.1f}%)")

        a = A[np.ix_(p1, p2)]
        r = score(a)
        r['p1'] = tuple(p1)
        r['p2'] = tuple(p2)
        results.append(r)

    return results

def get_results(pth):
    df = pd.read_csv(pth, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')
    A = df.to_numpy()

    output_file = pth.split('\\')[-1].split("/")[-1].split('.')[0] + '.csv'

    (m,n) = A.shape

    if max(m,n) < 10:
        print("doing a full eval")
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