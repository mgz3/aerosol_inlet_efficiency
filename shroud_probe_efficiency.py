from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp

# DIMENSIONES [m]
d_shroud = 20*1e-3
d_inlet = 6*1e-3
d_inlet_out = 8*1e-3
L_offset= 30*1e-3
L_shroud = 80*1e-3
L_inlet=250*1e-3

# PARAMETROS FLUIDO
level = 500 # altura en metros
atmos = ATMOSPHERE_1976(level)
U0 = 25 #m/s
U_U0 = 0.4
U_shroud = U0 * U_U0
U_inlet = U_shroud
Re_shroud = core.Reynolds(U0,d_shroud,atmos.rho,atmos.mu)
Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)

# PARTICULAS
rho_p = 1000 #kg/m**3    SUPOSICION
diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2])
diametros_micro = diametros_micro*10**-6



# EFICIENCIA ASPIRACION
def n_asp(U_U0,dp,V,diam):
    stokes = core.Stokes_number(V,dp,diam,rho_p,atmos.mu)
    n_aspiracion =  1 + ((1/U_U0)-1)*(1-(1+stokes*(2+0.617*U_U0))**-1) #valido para 0.005 < stokes < 10
    return n_aspiracion

EFICIENCIAS_SHROUD = []
EFICIENCIAS_INLET = []
EFICIENCIAS = []
for i in diametros_micro:
    EFICIENCIAS_SHROUD.append(n_asp(U_U0,i,U0,d_shroud))
    EFICIENCIAS_INLET.append(n_asp(0.8,i,U_inlet,d_inlet))

for i in range(len(EFICIENCIAS_SHROUD)):
    EFICIENCIAS.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(diametros_micro, EFICIENCIAS,'b')
plt.xlabel(r'Diametro $\mu$m', fontsize=14)
plt.ylabel(r'Eficiencia $\eta$', fontsize=14)
plt.title('Eficiencia de muestreo en funcion de diametro de particula',fontsize=16)
ax.grid()
plt.show()


# # transmision
# def n_trans(Stks,u0_u):
#     n_transport_inert = (1 + (u0_u - 1) / (1 + 2.66 * Stks ** (-2 / 3))) / (1 + (u0_u - 1) / (1 + 0.418 / Stks))
#     return n_transport_inert
