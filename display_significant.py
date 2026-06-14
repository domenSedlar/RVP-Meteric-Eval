import os
import pandas as pd
from display_heatmap import display_heatmap
import numpy as np
import ast

pths = [".\Matrices\small_random.tsv", ".\Matrices\large_random.tsv", ".\Matrices\GDS5600\\100.tsv"]

res_files = [os.path.join("./Results/", pth.split('\\')[-1].split("/")[-1].split('.')[0] + '.csv') for pth in pths]


metric_cols = ["ME4", "MI4", "MS4", "WV2", "WV3"]


for pth, res_pth in zip(pths, res_files):
    df = pd.read_csv(pth, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')
    A = df.to_numpy()
    df = pd.read_csv(res_pth, converters={c: ast.literal_eval for c in ['p1', 'p2']})
    print(df.columns)
    print(type(df['p1'].iloc[0]))
    df['MS4'] = -df['MS4']
    
    topk = {}
    k = 25
    tops = [set(df.nlargest(k, m)[['p1', 'p2']].apply(tuple, axis=1)) for m in metric_cols]
    print(tops)
    #print(tops)
    bottoms = [set(df.nsmallest(k, m)[['p1', 'p2']].apply(tuple, axis=1)) for m in metric_cols]
    
    top = tops[0]
    bottom = bottoms[0]
    
    for i in tops:
        #print(i)
        top = top.intersection(i)
    print(top)
    print("\n")
    for i in bottoms:
#        print(i)
        bottom = bottom.intersection(i)
   
    for i, (a,b) in enumerate(top):
        if i > 2:
            break
        display_heatmap(A[np.ix_(list(a), list(b))], title=pth.split('\\')[-1].split("/")[-1].split('.')[0] + " good permutation")

    for i, (a,b) in enumerate(bottom):
        if i > 2:
            break
        display_heatmap(A[np.ix_(list(a), list(b))], title=pth.split('\\')[-1].split("/")[-1].split('.')[0] + " bad permutation")