import numpy as np

small = np.random.rand(5,5)
large = np.random.rand(100,100)

np.savetxt('./Matrices/small_random.tsv', small, delimiter='\t')
np.savetxt('./Matrices/large_random.tsv',large, delimiter='\t')