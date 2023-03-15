'''
Four-well entanglement entropy
By: Daniel S Grun
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from numba import jit
#import time
from tqdm import tqdm

#start = time.time()

# Delta function

def d(x,y):
    if x == y:
        ans = 1
    else:
        ans = 0
    return ans


# State measurement collapse function

def meas(vec):

  '''
  Python function developed to employ state collapse in a
  measurement protocol.

  How to use it: simply call the function as meas(vec), where "vec" is the
  vector representation of the quantum state.

  Result: np.array([prob, vcol])
      -- prob: array with the two probabilities of measurement;
      -- vcol: array with the two possible collapsed represented states.
  '''

  coef03 = coefm3 = 0.
  psi03 = np.zeros((dim), dtype=complex)
  psim3 = np.zeros((dim), dtype=complex)

  for i in np.where(ets3==0)[0]:
    coef03 += np.abs(vec[i])**2
    psi03[i] = vec[i]

  for i in np.where(ets3==M)[0]:
    coefm3 += np.abs(vec[i])**2
    psim3[i] = vec[i]

  psi03 = psi03 / np.sqrt(coef03)
  psim3 = psim3 / np.sqrt(coefm3)

  return [np.array([coef03, coefm3]),
          np.array([psi03, psim3])]

# System / hamiltonian parameters

plt.close('all')

#n = input("Number of bosons: \n")

P = 11
M = 4

n = M + P

#mu2 = np.linspace(0,0.1,50)

#J = 12.914
#U = 10.776
#mu = 17.63

J = 10.707
U = 10.759
mu = 18.73

#J = 1
#U = 2

#J = 1
#U = 8
#mu = 10

lamb1 = J*J/(16*U) * (P/(M-P+1.) - (P+2.)/(M-P-1.))
lamb2 = J*J/(16*U) * ((M+2.)/(M-P+1.) - M/(M-P-1.))
lamb3 = J*J/(16*U) * (1./(M-P+1.) - 1./(M-P-1.))

tm = 2*np.pi*U/(J**2)*((M-P)**2 - 1)

tf = 2*tm # final time for evolution

#theta = np.linspace(0.,np.pi/P,50)

#tb = theta / mu

#dt = tb[1] - tb[0]

#tt = np.linspace(0, 2*tm, 1000)

s = 0
ss = 0

# Hilbert Space dimension

dim = np.int((n+3)*(n+2)*(n+1)/6.)
dim = dim

# Hamiltonian and initial state array definition

Heff = np.zeros((dim,dim))

H1 = np.zeros((dim,dim))
psi0 = np.zeros((dim),dtype=complex)

# Time-evolution operator matrix setup

contador = 0
cont = 0

def H(mu2):
  s = 0
  for i1 in range(0,n+1):
    for j1 in range(0,n-i1+1):
      for k1 in range(0,n-i1-j1+1):
        ss = 0
        for i2 in range(0,n+1):
          for j2 in range(0,n-i2+1):
            for k2 in range(0,n-i2-j2+1):
              l1 = n - j1 - i1 - k1
              l2 = n - j2 - i2 - k2

              H1[ss,s] = ( (U*(i1+k1-j1-l1)**2 + mu2*(j1-l1))*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1) - J/2.*(
                        1.*(np.sqrt(i1*(j1+1.))*d(i2,i1-1)*d(j2,j1+1) +
                                np.sqrt(j1*(i1+1.))*d(i2,i1+1)*d(j2,j1-1))*d(k2,k1)*d(l2,l1) +
                        1.*(np.sqrt(j1*(k1+1.))*d(j2,j1-1)*d(k2,k1+1) +
                                np.sqrt(k1*(j1+1.))*d(j2,j1+1)*d(k2,k1-1))*d(i2,i1)*d(l2,l1) +
                        1.*(np.sqrt(k1*(l1+1.))*d(k2,k1-1)*d(l2,l1+1) +
                                np.sqrt(l1*(k1+1.))*d(k2,k1+1)*d(l2,l1-1))*d(i2,i1)*d(j2,j1) +
                        1.*(np.sqrt(i1*(l1+1.))*d(i2,i1-1)*d(l2,l1+1) +
                                np.sqrt(l1*(i1+1.))*d(i2,i1+1)*d(l2,l1-1))*d(j2,j1)*d(k2,k1)) )

              ss += 1
        s += 1
  return H1

uint = la.expm(-1j*H(0)*tm)
#ubreak = la.expm(-1j*H(mu)*dt)
#u20 = la.expm(1j*H(0)*dt/2.)

(evl,evc) = la.eigh(H(0))

# Initial and NOON states representation

F = np.zeros((dim,3), dtype=int)
s = 0

for i in range(0,n+1):
  for j in range(0,n+1-i):
    for k in range(0,n+1-i-j):

      F[s,0] = i
      F[s,1] = j
      F[s,2] = k

      s += 1

X = np.zeros((n+1,n+1,dim), dtype=int)

for i in range(0,dim):
  X[F[i][0], F[i][1], F[i][2]] = i

s0 = s1 = s2 = s3 = 0
(ets1, ets2, ets3, ets4) = (np.zeros((dim)),
                            np.zeros((dim)),
                            np.zeros((dim)),
                            np.zeros((dim))) # for state-collapse protocol

s = 0

for i in range(0,n+1):
  for j in range(0,n+1-i):
    for k in range(0,n+1-i-j):
      l = n - i - j - k

      ets1[s] = i
      ets2[s] = j
      ets3[s] = k
      ets4[s] = l

      s += 1

s0, s1, s2, s3 = (X[M,P,0],
                  X[M,0,0],
                  X[0,P,M],
                  X[0,0,M])

# Number matrix operator

Ni = [np.diag(ets1),
      np.diag(ets2),
      np.diag(ets3),
      np.diag(ets4)]

psi0[s0] = 1.0 # "pure" initial state component

psitm = uint @ psi0

tt = np.linspace(0,tf,500)

von = np.zeros((len(tt)))

#nvon = np.zeros((len(theta)))

print("Defining beta's")


@jit(nopython=True) #, parallel=True
def beta(t,s):
  bt = 0
  for i in range(0,dim):
    for j in range(0,dim):
      for r in range(0,n+1-s):
        for w in range(0,n+1-r-s):
          bt = bt + ( evc[s0,i]*evc[s0,j]
                     *evc[X[s,r,n-s-r-w],i]*evc[X[s,r,n-s-r-w],j]
                     *np.cos((evl[i]-evl[j])*tt[t]) )
  return bt

'''
def noonbeta(th,s):
  noon = np.zeros((dim), dtype=complex)
  noon[s0] = np.cos(P*th)
  noon[s1] = np.sin(P*th)
  nbt = sum(sum([np.dot(noon,evc[:,i])*np.dot(evc[:,j],noon)
           *evc[X[s,r,n-s-r-w],i]*evc[X[s,r,n-s-r-w],j] for j in range(0,dim)
           for i in range(0,dim)]) for r in range(0,n+1-s)
           for w in range(0,n+1-r-s))
  return nbt
'''

print("Calculating Von-Neumann entropy")

for t in tqdm(range(0,len(tt))):
  von[t] = sum([- beta(t,s)*np.log2(beta(t,s)) for s in range(0,n+1)])

np.savetxt('entropy-U{0:1d}-J{1:1d}-{2:1d}{3:1d}-python.txt'.format(int(U),int(J),M,P),np.c_[tt,von])
