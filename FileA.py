import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh, norm,svd
import scipy
import scipy.io
np.set_printoptions(threshold=np.inf)
import itertools
import multiprocessing
import tables
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# module = getattr(__import__("SVD_Trader"),'SVD')
# Create a sample matrix (you can replace this with youru own data)
def UExp(U, NY, NDays):


  UExp = np.zeros((U.shape[0] * (NY + 1), U.shape[1]))
  for Yr in range(1, NY + 1):
    UExp[NDays * (Yr - 1) + np.arange(1, NDays + 1), :] = U + np.ones((NDays, 1)) * (Yr - 1) * (U[-1, :] - U[0, :])
  return UExp



df_1 = pd.read_csv('TSLA.csv')
df_2 = pd.read_csv('RIVN.csv')
print('finished')
# print(df_1.shape)
# print(df_2.shape)

#mat = scipy.io.loadmat('Data_0511.mat')
#C = np.array(mat['YearDataMat'])
#NDays = len(C)

# Perform SVD on the matrix A
B = pd.concat([df_2,df_1],axis=1)
C = B.to_numpy()

u, s, VT = svd(C, full_matrices=False)
bruh = np.array(u)
#bbruh = UExp(bruh,4,NDays)
U = pd.DataFrame(bruh)


#print(U)
#breakpoint()
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
plt.plot(X, y)
plt.plot(X, z)
plt.plot(X, a)
plt.plot(X, b)

plt.show()
print(scipy.stats.pearsonr(z, b))

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