'''
Four-well energy bands
By: Nicholas J. Lopes
'''


import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

def d(x,y):
    if x == y:
        ans = 1
    else:
        ans = 0
    return ans

n=15
J=1.0


dim = int((n+3.)*(n+2.)*(n+1.)/6.)

U = np.array(np.arange(0,10,0.01))
H = np.zeros((len(U),dim,dim))
E = np.zeros((len(U),dim))

s=0
ss=0

for i in tqdm(range(0,len(U))):
    for i1 in range(0,n+1):
        for j1 in range(0,n+1-i1):
            for k1 in range(0,n+1-i1-j1):
                l1 = n - i1 - j1 - k1
                ss = 0
                for i2 in range(0,n+1):
                    for j2 in range(0,n+1-i2):
                        for k2 in range(0,n+1-i2-j2):
                            l2 = n - i2 - j2 - k2
                            H[i,ss,s] = (U[i]*(i1-j1+k1-l1)**2*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
                                         +J/2.*np.sqrt(j1*(i1+1.))*d(i2,i1+1)*d(j2,j1-1)*d(k2,k1)*d(l2,l1)
                                         +J/2.*np.sqrt(l1*(i1+1.))*d(i2,i1+1)*d(j2,j1)*d(k2,k1)*d(l2,l1-1)
                                         +J/2.*np.sqrt(j1*(k1+1.))*d(i2,i1)*d(j2,j1-1)*d(k2,k1+1)*d(l2,l1)
                                         +J/2.*np.sqrt(l1*(k1+1.))*d(i2,i1)*d(j2,j1)*d(k2,k1+1)*d(l2,l1-1)
                                         +J/2.*np.sqrt(i1*(j1+1.))*d(i2,i1-1)*d(j2,j1+1)*d(k2,k1)*d(l2,l1)
                                         +J/2.*np.sqrt(k1*(j1+1.))*d(i2,i1)*d(j2,j1+1)*d(k2,k1-1)*d(l2,l1)
                                         +J/2.*np.sqrt(i1*(l1+1.))*d(i2,i1-1)*d(j2,j1)*d(k2,k1)*d(l2,l1+1)
                                         +J/2.*np.sqrt(k1*(l1+1.))*d(i2,i1)*d(j2,j1)*d(k2,k1-1)*d(l2,l1+1))
                            ss+=1
                s+=1
    s = 0

    (eigvals,eigvec) = la.eigh(H[i,:,:])
    E[i,:] = eigvals

plt.figure(figsize=(10,8))

plt.plot(U,E)
plt.ylim(-16,100)
plt.xlim(0,9.9)
plt.xlabel("U/J")
plt.ylabel("E/J")
plt.show()
plt.savefig('bandas.png')
