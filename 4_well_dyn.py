'''
Four-well tunneling dynamics
By: Nicholas J. Lopes
'''


import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

def d(x,y):					            #Funcao delta
    if x == y:
        ans = 1
    else:
        ans = 0
    return ans


U = input("Enter U:")                   #Parâmetro de acoplamento
U = float(U)

n = input("Enter N:")                   #Número de particulas no sistema
n = int(n)


dim = int((n+3.)*(n+2.)*(n+1.)/6.)      #Dimensão do espaço de Hilbert

s = 0

for i1 in range(0,n+1):                 #Printa os possíveis estados iniciais para o sistema
    for j1 in range(0,n+1-i1):
        for k1 in range (0, n+1-i1-j1):
            l1 = n - i1 - j1 - k1
            print(s,"|",i1,j1,k1,l1,">")
            s+=1

i0 = input("Initial state: ")
i0 = int(i0)                            #Seleção do estado inicial do sistema

P = input("Enter P:")                   #Número de patículas nos poços 2 e 4
P = int(P)

psi0 = np.zeros((dim),dtype=complex)
psi0[i0] = 1.

J=1.

tm = (np.pi/(2*(J**2/(4*U*((n-2*P)**2-1)))))

a1 = a2 = 1./np.sqrt(2.)
b1 = b2 = 1./np.sqrt(2.)

H = np.zeros((dim,dim))
N1 = np.zeros((dim,dim))
N2 = np.zeros((dim,dim))
N3 = np.zeros((dim,dim))
N4 = np.zeros((dim,dim))

s=0
ss=0


for i1 in range(0,n+1):                 #Dinâmica do Hamiltoniano do sistema
    for j1 in range(0,n+1-i1):
        for k1 in range(0,n+1-i1-j1):
            l1 = n - i1 - j1 - k1
            ss = 0
            for i2 in range(0,n+1):
                for j2 in range(0,n+1-i2):
                    for k2 in range(0,n+1-i2-j2):
                        l2 = n - i2 - j2 - k2
                        H[s,ss] = (U*(i1-j1+k1-l1)**2*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
                                     +a1*b1*np.sqrt(j1*(i1+1.))*d(i2,i1+1)*d(j2,j1-1)*d(k2,k1)*d(l2,l1)
                                     +a1*b2*np.sqrt(l1*(i1+1.))*d(i2,i1+1)*d(j2,j1)*d(k2,k1)*d(l2,l1-1)
                                     +a2*b1*np.sqrt(j1*(k1+1.))*d(i2,i1)*d(j2,j1-1)*d(k2,k1+1)*d(l2,l1)
                                     +a2*b2*np.sqrt(l1*(k1+1.))*d(i2,i1)*d(j2,j1)*d(k2,k1+1)*d(l2,l1-1)
                                     +a1*b1*np.sqrt(i1*(j1+1.))*d(i2,i1-1)*d(j2,j1+1)*d(k2,k1)*d(l2,l1)
                                     +a2*b1*np.sqrt(k1*(j1+1.))*d(i2,i1)*d(j2,j1+1)*d(k2,k1-1)*d(l2,l1)
                                     +a1*b2*np.sqrt(i1*(l1+1.))*d(i2,i1-1)*d(j2,j1)*d(k2,k1)*d(l2,l1+1)
                                     +a2*b2*np.sqrt(k1*(l1+1.))*d(i2,i1)*d(j2,j1)*d(k2,k1-1)*d(l2,l1+1))
                        ss+=1
            s+=1

s=0
for i1 in range(0,n+1):
    for j1 in range(0,n+1-i1):
        for k1 in range(0,n+1-i1-j1):
            l1 = n - i1 - j1 -k1
            ss = 0
            for i2 in range(0,n+1):
                for j2 in range(0,n+1-i2):
                    for k2 in range(0,n+1-i2-j2):
                        l2 = n - i2 - j2 -k2
                        N1[ss,s] = i1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
                        N2[ss,s] = j1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
                        N3[ss,s] = k1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
                        N4[ss,s] = l1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
                        ss += 1
            s+=1

(evl,evc) = la.eigh(H)

tf=2*tm

time = np.linspace(0,2*tm,3000)

psit = np.zeros((len(time),dim),dtype=complex)
psitt = 0


for t in tqdm(range(0,len(time))):
    for i in range(0,dim):
        psitt = np.dot(np.conj(evc[:,i]),psi0)*np.exp(-1j*evl[i]*time[t])
        psit[t,:] += psitt*evc[:,i]


nat = np.zeros((len(time)))
nbt = np.zeros((len(time)))
nct = np.zeros((len(time)))
ndt = np.zeros((len(time)))

for t in tqdm(range(0,len(time))):
    nat[t] = np.dot(np.conj(psit[t,:]),N1 @ psit[t,:])
    nbt[t] = np.dot(np.conj(psit[t,:]),N2 @ psit[t,:])
    nct[t] = np.dot(np.conj(psit[t,:]),N3 @ psit[t,:])
    ndt[t] = np.dot(np.conj(psit[t,:]),N4 @ psit[t,:])

plt.plot(time, nat, color='r', alpha=1., lw=0.8, label="$<N_1>$")
plt.plot(time, nbt, color='g', alpha=1., lw=0.8, label="$<N_2>$")
plt.plot(time, nct, color='b', alpha=1., lw=0.8, linestyle='dashed', label="$<N_3>$")
plt.plot(time, ndt, color='orange', alpha=1., lw=0.8, linestyle='dashed', label="$<N_4>$")
plt.legend(loc=1)
plt.xlabel("t")
plt.ylabel("$<N_i>$")
plt.ylim(0,n+1.0)
plt.xlim(0,tf)

#plt.show()

plt.savefig('D123.png')
plt.close
