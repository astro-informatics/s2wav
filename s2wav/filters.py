import numpy as np
from tiling import *
from dimension_helper_functions import *


def tiling_phi2(L:int, lam:float) -> float:

    n = 300
    J = j_max(L, lam)
    print("J max is ", J)

    kappanorm = part_scaling_fn(1/lam, 1.0, n, lam)
    print("Found kappanorm")
    phi2 = np.zeros((J + 2) * L)
    

    for j in range(J+2):
        for l in range(L):
            if l < lam**(j-1):
                phi2[l + j * L] = 1
            elif l > lam**j:
                phi2[l + j * L] = 0
            else:
                phi2[l + j * L] =  part_scaling_fn(l/lam**j, 1.0, n, lam)/ kappanorm
            

    return phi2


def tiling_axisym(L: int, lam:float, J_min: int):

    J = j_max(L, lam)

    previoustemp = 0.0
    print("Finding phi2")
    phi2 = tiling_phi2(L, lam)
    kappa = np.zeros((J + 1) * L)
    kappa0 = np.zeros(L)
    print("Finding kappa0")
    for l in range(L):
      kappa0[l] = np.sqrt(phi2[l + J_min * L])
    
    for j in range(J_min, J+1):
        for l in range(L):
            diff = phi2[l + (j + 1) * L] - phi2[l + j * L] 
            #check if sqrt is defined
            if diff < 0:
              kappa[l + j * L] = previoustemp;
            else:
              temp = np.sqrt(diff)
              kappa[l + j * L] = temp
              previoustemp = temp

        for l in range(L):
            kappa[l + j * L] = kappa[l + j * L - 1]
          
    return kappa, kappa0

print(tiling_axisym(10, 2, 2))