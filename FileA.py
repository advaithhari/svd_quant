import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh, norm,svd
import scipy
np.set_printoptions(threshold=np.inf)
import itertools
import multiprocessing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

A = np.array([[-4,0],
              [2,1],
              [3,-2]])
U, E, VT = svd(A)
print(U)
# breakpoint()
column_rank = np.linalg.matrix_rank(U, tol=None)
print(column_rank)
module = getattr(__import__("SVD_Trader"),'SVD')
# Create a sample matrix (you can replace this with your own data)
print('processing...')

df_1 = module('TSLA')
df_2 = module('RIVN')
print('finished')
# print(df_1.shape)
# print(df_2.shape)
print(pd.concat([df_1,df_2],axis=1).shape)
# Perform SVD on the matrix A
A = pd.concat([df_2,df_1],axis=1)
print(A)
print(A.values)
U, s, VT = svd(A, full_matrices=False)
print(U)
breakpoint()
U, E, VT = svd(A)
print(U)
X =[x for x in range(len(U))]
print(len(U[0]))
print((len(U)))
# for col in range(len(U[0])):
y = list()
z = list()
a = list()
b = list()
for row in range(len(U)):
    y.append(U[0][row])
    z.append(U[1][row])
    a.append(U[2][row])
    b.append(U[3][row])
    # print(col)
col1 = [f-e for (e, f) in zip(y, z)]
col2 = [f-e for (e, f) in zip(a, b)]
plt.plot(X, col1)
plt.plot(X, col2)
plt.show()
print(scipy.stats.pearsonr(col1, col2))

# print(A)
# ev, V = eigh(A.T@A)
# print(V)
# print(V.shape)
# u = list()
# for i in range(A.shape[0]):
#     u.append(A@V[:,i]/norm(A@V[:,i]))
#
# U = np.array(u[::-1]).T

# U is the left singular vectors
# S is the singular values (a 1-D array)
# VT is the right singular vectors (transposed)

# Print the results
# print("U (left singular vectors):\n", U)
# diagonal_values = np.diagonal(U)
# print("Diagonal values:", diagonal_values)
# plt.plot([x for x in range(len(diagonal_values))],diagonal_values)
# plt.show()
# print("S (singular values):\n", S)
# print("VT (right singular vectors transposed):\n", VT)