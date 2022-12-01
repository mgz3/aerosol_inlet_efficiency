from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp, pi, cos, sqrt, asin

# DIMENSIONES [m]
d_shroud = 20*1e-3
d_inlet = 6*1e-3
d_inlet_out = 8*1e-3
L_offset= 30*1e-3
L_shroud = 80*1e-3
L_inlet=170*1e-3

# PARAMETROS FLUIDO
level = 100 # altura en metros
atmos = ATMOSPHERE_1976(level)
U0 = 20 #m/s
U_U0 = 0.4
U_shroud = U0 * U_U0
Q_shroud =U_shroud*(pi*(d_shroud)**2/4)     #m**3/s  caudal volumetrico
Re_shroud = core.Reynolds(U0,d_shroud,atmos.rho,atmos.mu)
Q_inlet = 2    #L/min  caudal volumetrico estipulado por Renard
Q_inlet = Q_inlet/60000    #m**3/s
U_inlet = 4*Q_inlet/(d_inlet**2*pi)
Uinlet_Ushroud = U_inlet/U_shroud
Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)

# PARTICULAS
rho_p = 1000 #kg/m**3    SUPOSICION
diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2])
diametros_micro = diametros_micro*10**-6


# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA MUESTREO
# aspiracion
def n_asp(u_u0,dp,V,diam):
    stokes = core.Stokes_number(V,dp,diam,rho_p,atmos.mu)
    n_aspiracion =  1 + ((1/u_u0)-1)*(1-(1+stokes*(2+0.617*u_u0))**-1) #valido para 0.005 < stokes < 10
    return n_aspiracion


# transmision
def n_trans(u_u0,dp,V,diam):
    u0_u = 1/u_u0
    stokes = core.Stokes_number(V, dp, diam, rho_p, atmos.mu)
    n_transport_inert = (1 + (u0_u - 1) / (1 + 2.66 / stokes ** (2 / 3))) / (1 + (u0_u - 1) / (1 + 0.418 / stokes))
    return n_transport_inert

# factor correccion Gong et al.
def factor_corr(u_u0,dp,V,diam):
    stokes = core.Stokes_number(V, dp, diam, rho_p, atmos.mu) #stokes del shroud
    u0_u = 1/u_u0
    if not 8900 <= Re_shroud <= 54000:
        print('El valor del reynolds esta fuera del rango especificado por Gong.')
    factor = 1 - (u0_u-1)*0.861*stokes/(stokes*(2.34+0.939*(u0_u-1))+1)
    return factor

# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA TRANSPORTE
# .............................................
# parametros auxiliares
k_boltzman = 1.38*10**-23 #N.m/K

def tau_rel(rho_p, dp, mu, P, T):
    return rho_p * dp**2 * Cc(dp, P, T) / (18 * mu)
    # return rho_p * dp**2 * Cc_2(dp, P, T) / (18 * mu)

def Kn(dp, P, T):
    P = P * 10 ** -3
    lambda_r = 0.0664 * (101 / P) * (T / 293) * ((1 + 110 / 293) / (1 + 110 / T))
    lambda_r = lambda_r *10**-6
    return lambda_r
    # return 2 * LAMBDA(P, T) / dp

def Cc(dp, P, T):
    # alpha = 1.142
    # beta = 0.558
    # gamma = 0.999
    alpha = 1.207
    beta = 0.440
    gamma = 0.596
    # print(dp)
    # print(P)
    # print(T)
    # print(Kn(dp,P,T))
    return 1 + Kn(dp, P, T) * (alpha + beta * exp(-gamma / Kn(dp, P, T)))
    # P = P*10**-3
    # cc2 = 1 + 1/(P*dp)*(15.60+7*exp(-0.059*P*dp))
    # return cc2


def Cc_2(dp, P, T):
    dp_micro = dp * 10 ** 6
    p_kpa = P * 10 ** -3
    alpha = 15.6
    beta = 7
    gamma = 0.059
    return 1 + 1 / (p_kpa * dp_micro) * (alpha + beta * exp(-gamma / (p_kpa * dp_micro)))


# .............................................

# sedimentacion
def n_sedim(d,L,rho_p,dp,mu,P,T,Q,V,re,ang=0):
    """
    Args:
        d: diametro interno del tubo
        L: largo del tubo
        rho_p: densidad de la particula
        dp: diametro de la particula
        mu: viscosidad dinamica
        P: presion atmosferica
        T: temperatura
        Q: caudal volumetrico
        ang: angulo inclinacion de la sonda
    Returns: rendimiento difusivo
    """
    Vts = tau_rel(rho_p,dp,mu,P,T)*9.81

    # eta = 3/4*L/d*Vts/V
    # rendimiento = 1-2/pi*(2*eta*sqrt(1-eta**(2/3))-eta**(1/3)*sqrt(1-eta**(2/3))+asin(eta**(1/3)))

    # stokes = core.Stokes_number(V, dp, d, rho_p, atmos.mu)
    # Z = L*Vts/(d*V)
    # K = sqrt(Z*stokes)*re**(-1/4)
    # rendimiento = exp(-4.7*K**0.75)
    # return rendimiento
    valor = exp(-1*(d*L*Vts*cos(ang*pi/180))/Q)
    print('este valor: ',valor)
    return valor


# difusion
def n_dif(dp,P,T,L,Q,mu,re,rho_f):
    """

    Args:
        dp: diametro de particula
        P: presion atmosferica
        T: temperatura
        L: largo del tubo
        Q: caudal volumetrico
        mu: viscosidad dinamica
        rho_d: densidad de la particula
        re: reynolds
        rho_f: densidad del aire

    Returns: rendimiento difusivo

    """
    # B = Cc(dp,P,T)/(3*pi*mu*dp)
    B = Cc_2(dp,P,T)/(3*pi*mu*dp)
    D = k_boltzman*T*B
    zita = pi *D*L/Q
    Sc=mu/(rho_f*D)
    Sh = 0.0118*re**(7/8)*Sc**(1/3)
    n = exp(-zita * Sh)
    return n

def n_turb_inert(d,L,Q,re,V,dp):
    """

    Args:
        d: diametro interno del tubo
        L:largo del tubo
        Q: caudal volumetrico
        Stks: numero de stokes
        re: numero de reynolds
        u: velocidad en el tramo

    Returns: rendimiento inercial turbulento

    """
    stokes = core.Stokes_number(V, dp, d, rho_p, atmos.mu)
    print(dp, stokes)
    tau_mas = 0.0395*stokes*re**(3/4)
    if tau_mas > 12.9:
        V_mas = 0.1
    else:
        V_mas = 6*10**-4*tau_mas**2+2*10**-8*re
    Vt = V_mas*V/(5.03*re**(1/8))
    # Vt = (6*10**-4*(0.0395*stokes*re**(3/4))**2+re*2*10**(-8))/(5.03*re**(1/8))*V
    return exp(-(pi*d*L*Vt)/(Q))
# ----------------------------------------------------------------------------------------------------------------------
EFICIENCIAS_SHROUD = []
shroud_eff_asp = []
shroud_eff_transm = []
shroud_eff_sed = []
shroud_eff_diff = []
shroud_eff_turb = []
EFICIENCIAS_INLET = []
inlet_eff_asp = []
inlet_eff_transm = []
inlet_eff_sed = []
inlet_eff_diff = []
inlet_eff_turb = []
EFICIENCIAS_TOTAL = []
FACTOR = []
print('SHROUD')
for dp in diametros_micro:
    # FACTOR
    FACTOR.append(factor_corr(U_U0,dp,U0,d_shroud))
    # ..............................................
    # SHROUD
    shroud_eff_asp.append(n_asp(U_U0,dp,U0,d_shroud))
    shroud_eff_transm.append(n_trans(U_U0,dp,U0,d_shroud))
    shroud_eff_diff.append(n_dif(dp,atmos.P,atmos.T,L_offset,Q_shroud,atmos.mu,Re_shroud,atmos.rho))
    shroud_eff_sed.append(n_sedim(d_shroud,L_offset,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_shroud,U_shroud,Re_shroud))
    shroud_eff_turb.append(n_turb_inert(d_shroud,L_offset,Q_shroud,Re_shroud,U_shroud,dp))

print('-----------------------------------')
print('INLET')
for dp in diametros_micro:
    # ..............................................
    # INLET
    inlet_eff_asp.append(n_asp(Uinlet_Ushroud,dp,U_inlet,d_inlet))
    inlet_eff_transm.append(n_trans(Uinlet_Ushroud,dp,U_inlet,d_inlet))
    inlet_eff_diff.append(n_dif(dp,atmos.P,atmos.T,L_inlet,Q_shroud,atmos.mu,Re_inlet,atmos.rho))
    inlet_eff_sed.append(n_sedim(d_inlet,L_inlet,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_inlet,U_inlet,Re_inlet))
    inlet_eff_turb.append(n_turb_inert(d_inlet,L_inlet,Q_inlet,Re_inlet,U_inlet,dp))
    # inlet_eff_turb.append(n_turb_inert(d_inlet,L_inlet,Q_inlet,Re_inlet,U_inlet,dp))





for i in range(len(diametros_micro)):
    # EFICIENCIAS_SHROUD.append(shroud_eff_asp[i]*shroud_eff_transm[i]*shroud_eff_diff[i])
    # EFICIENCIAS_INLET.append(inlet_eff_asp[i]*inlet_eff_transm[i]*inlet_eff_diff[i])
    EFICIENCIAS_SHROUD.append(shroud_eff_asp[i]*shroud_eff_transm[i]*shroud_eff_sed[i]*shroud_eff_diff[i])
    EFICIENCIAS_INLET.append(inlet_eff_asp[i]*inlet_eff_transm[i]*inlet_eff_sed[i]*inlet_eff_diff[i])
    EFICIENCIAS_TOTAL.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i]*FACTOR[i])
    # EFICIENCIAS_TOTAL.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i])


import matplotlib.pyplot as plt
fig1, axs = plt.subplots(3,2,constrained_layout = True)
for ax in axs.flat:
    ax.grid()
    ax.set(xlabel=r'Diametro $\mu$m', ylabel=r'Eficiencia $\eta$')


ax1 = axs[0, 0] # sampling shroud
ax1.set_title('sampling shroud')
ax2 = axs[0, 1] # sampling inlet
ax2.set_title('sampling inlet')
ax3 = axs[1, 0] # transport shroud
ax3.set_title('transport shroud')
ax4 = axs[1, 1] # transport inlet
ax4.set_title('transport inlet')
ax5 = axs[2, 0] # total shroud
ax5.set_title('total shroud')
ax6 = axs[2, 1] # total inlet
ax6.set_title('total inlet')


diametros_micro = diametros_micro/10**-6

def plot_eff(ax,diametros,eficiencias,color,label):
    ax.plot(diametros, eficiencias,color,label=label)
    ax.legend()


# SHROUD
plot_eff(ax1,diametros_micro,shroud_eff_asp,'b','asp shroud')
plot_eff(ax1,diametros_micro,shroud_eff_transm,'r','transm shroud')
plot_eff(ax3,diametros_micro,shroud_eff_diff,'r','difusion')
plot_eff(ax3,diametros_micro,shroud_eff_turb,'g','turbulento')
plot_eff(ax3,diametros_micro,shroud_eff_sed,'b','sedimentacion')
plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD,'k','TOTAL SHROUD')
plot_eff(ax5,diametros_micro,FACTOR,'g','FACTOR')
# INLET
plot_eff(ax2,diametros_micro,inlet_eff_asp,'b','asp inlet')
plot_eff(ax2,diametros_micro,inlet_eff_transm,'r','transm inlet')
plot_eff(ax4,diametros_micro,inlet_eff_diff,'r','difusion')
plot_eff(ax4,diametros_micro,inlet_eff_turb,'g','turbulento')
plot_eff(ax4,diametros_micro,inlet_eff_sed,'b','sedimentacion')
plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET,'k','TOTAL INLET')
plot_eff(ax6,diametros_micro,FACTOR,'g','FACTOR')


fig2, ax = plt.subplots(1,1)

ax.plot(diametros_micro, EFICIENCIAS_TOTAL,label='EFICIENCIA TOTAL')
ax.plot(diametros_micro, EFICIENCIAS_INLET,label='EFICIENCIAS_INLET')
ax.plot(diametros_micro, EFICIENCIAS_SHROUD,label='EFICIENCIAS_SHROUD')
ax.plot(diametros_micro, FACTOR,label='FACTOR')
ax.legend()
ax.grid()
ax.set_xlabel(r'Particle Diameter, dp (\textmu m)')
ax.set_ylabel(r'Efficiency $\eta$')

# fig1.show()
# fig2.show()
plt.show()
print('done')


# plt.show()

