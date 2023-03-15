def ent_plotter(U, J, M, P):
  '''

    Parameters
    ----------
    U : Float
        Hamiltonian interaction parameter.
    J : Float
        Hamiltonian tunneling parameter.
    M : Int
        Number of atoms in site 1 (for |Psi0> = |M,P,0,0> ).
    P : Int
        Number of atoms in site 2 (for |Psi0> = |M,P,0,0> ).
    mu : Float
        Breaking of integrability parameter.

    Returns
    -------
    Figure with the Von-Neumann entropy plotted against time.

  '''

  import matplotlib.pyplot as plt
  from math import pi
  import numpy as np

  t,von = np.loadtxt('entropy-U{0:1d}-J{1:1d}-{2:1d}{3:1d}-python.txt'
                     .format(int(U),int(J),M,P),
                     unpack=True)

  t,voneff = np.loadtxt('entropy-U{0:1d}-J{1:1d}-{2:1d}{3:1d}-eff-python.txt'
                     .format(int(U),int(J),M,P),
                     unpack=True)

  tm = 2*pi*U*((M-P)*(M-P) - 1)/(J*J)

  plt.figure(figsize=(8,6))

  plt.plot(t[:], von[:], label="$H$", color='royalblue', lw=2.0, alpha=0.7)
  plt.plot(t[:], voneff[:], label="$H_{eff}$", color='purple', lw=2.0, alpha=0.7,
           linestyle = "--")
  plt.xlim(0,2*tm)
  plt.ylim(0,2.5)
  plt.axhline(y=1.0, xmax=1, color='k', lw=1.5, linestyle=':', alpha=0.7)
  plt.axvline(x=tm, ymax=1, color='k', lw=1.5, linestyle=':', alpha=0.7)
  plt.legend(loc=4, fontsize=17)
  plt.xlabel("t (s)", fontsize=18)
  plt.ylabel("$S(\\rho_1)$", fontsize=18)
  plt.title("M = {0:1d}, P = {1:1d}, U = {2:2.3f}, J = {3:2.3f}"
            .format(M,P,U,J),
            fontsize=17)
  plt.xticks(ticks=[0, 12, 24, tm, 36, 48, 60],
             #ticks=[0, 15, 30, tm, 45, 60, 75],
             #ticks=[0, 8, 16, tm, 24, 32, 40],
             labels=['0', '12', '24', '$t_m$', '36', '48', '60'],
             #labels=['0','15', '30', '$t_m$', '45', '60', '75'],
             #labels=['0','8', '16', '$t_m$', '24', '32', '40'],
             fontsize=17)
  plt.yticks(fontsize=17)

  #plt.show()

  plt.savefig('entropy_HxHeff.png')
  plt.close


ent_plotter(10.759, 10.707, 4, 11)
