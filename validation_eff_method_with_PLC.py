import matplotlib.pyplot as plt
from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp, pi, cos, sqrt, asin
from scipy.optimize import curve_fit
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
def replace_comma_with_dot(x):
    try:
        return float(x.replace(",", "."))
    except:
        return x


wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/PLC/'
file = 'asp.csv'
df = pd.read_csv(wd + file)
df_asp = df.applymap(replace_comma_with_dot)
file = 'diff.csv'
df = pd.read_csv(wd + file)
df_diff = df.applymap(replace_comma_with_dot)
file = 'sampling.csv'
df = pd.read_csv(wd + file)
df_sampling = df.applymap(replace_comma_with_dot)
file = 'sedim.csv'
df = pd.read_csv(wd + file)
df_sedim = df.applymap(replace_comma_with_dot)

# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA MUESTREO
# aspiracion
def n_asp(u0_u,stokes):
    # n_aspiracion =  1 + (u0_u-1)*(1-(1+stokes*(2+0.617*(1/u0_u)))**-1) #valido para 0.005 < stokes < 2.03
    n_aspiracion =  1 + (u0_u-1)*(1-(1+3.77*stokes**(0.883))**-1) #valido para 0.005 < stokes < 10
    return n_aspiracion


# transmision
def n_trans(u0_u,stokes):
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
    tau = rho_p * dp**2 * Cc(dp, P, T) / (18 * mu)
    return tau

def Kn(dp, P, T):
    P = P * 10 ** -3
    lambda_r = 0.0664 * (101 / P) * (T / 293) * ((1 + 110 / 293) / (1 + 110 / T))
    lambda_r = lambda_r *10**-6
    kn = 2*lambda_r/dp
    return kn

def Cc(dp, P, T):
    P = P * 10 ** -3
    dp = dp/ 10 ** -6
    cc = 1 + 1*(15.60+7*np.exp(-0.059*P*dp))/(P*dp)
    return cc


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
    # print(Vts)
    eficiencia = np.exp(-d*L*Vts/Q)
    return eficiencia


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
    B = Cc(dp,P,T)/(3*pi*mu*dp)
    D = k_boltzman*T*B
    zita = pi *D*L/Q
    Sc=mu/(rho_f*D)
    Sh = 0.0118*re**(7/8)*Sc**(1/3)
    n = np.exp(-zita * Sh)
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
    return np.exp(-(pi*d*L*Vt)/(Q))

# DIMENSIONES [m]
d_shroud = 20*1e-3
d_inlet = 6*1e-3
d_inlet_out = 8*1e-3
L_offset= 30*1e-3
L_shroud = 80*1e-3
L_inlet=230*1e-3
# para hacer el analisis dividiendo en tramos
DOS_TRAMOS = True
L_inlet_1=15*1e-3 #largo del tramo 1
L_inlet_2=L_inlet-L_inlet_1 #largo del tramo 2
d_inlet_1= 6*1e-3
d_inlet_2=8.5*1e-3 #diametro del tramo 2


# PARAMETROS FLUIDO
level = 100 # altura en metros
atmos = ATMOSPHERE_1976(level)
U0 =2.2 #m/s
Q_inlet = 2    #L/min  caudal volumetrico estipulado por Renard
Q_inlet = Q_inlet/60000    #m**3/s
U_inlet = 4*Q_inlet/(d_inlet**2*pi)
Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)
# para hacer el analisis dividiendo en tramos
Re_inlet_1 = core.Reynolds(U_inlet,d_inlet_1,atmos.rho,atmos.mu)
Re_inlet_2 = core.Reynolds(U_inlet,d_inlet_2,atmos.rho,atmos.mu)

# PARTICULAS
rho_p = 1000 #kg/m**3    SUPOSICION  ESTANDAR
diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2,50])
diametros_micro = diametros_micro*10**-6


asp=[]
trans = []
samp = []
inlet = []
sed = []
turb = []
diff = []
particulas = np.linspace(0.1,50,300)*10**-6
U0_U = U0/U_inlet
for dp in particulas:
    stk = core.Stokes_number(U_inlet,dp,d_inlet,rho_p,atmos.mu)
    asp_val = n_asp(U0_U,stk)
    trans_val = n_trans(U0_U,stk)
    samp_val = asp_val * trans_val

    turb_val_1 = n_turb_inert(d_inlet_1,L_inlet_1,Q_inlet,Re_inlet,U_inlet,dp)
    sed_val_1 = n_sedim(d_inlet_1,L_inlet_1,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_inlet,U_inlet,Re_inlet)
    diff_val_1 = n_dif(dp,atmos.P,atmos.T,L_inlet_1,Q_inlet,atmos.mu,Re_inlet,atmos.rho)

    turb_val_2 = n_turb_inert(d_inlet_2,L_inlet_2,Q_inlet,Re_inlet,U_inlet,dp)
    sed_val_2 = n_sedim(d_inlet_2,L_inlet_2,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_inlet,U_inlet,Re_inlet)
    diff_val_2 = n_dif(dp,atmos.P,atmos.T,L_inlet_2,Q_inlet,atmos.mu,Re_inlet,atmos.rho)


    asp.append(asp_val)
    trans.append(trans_val)
    samp.append(samp_val)
    sed.append(sed_val_1*sed_val_2)
    turb.append(turb_val_1*turb_val_2)
    diff.append(diff_val_1*diff_val_2)
    inlet.append(samp_val*sed_val_1*sed_val_2*turb_val_1*turb_val_2*diff_val_1*diff_val_2)

particulas = np.linspace(0.1,50,300)

# plt.plot(particulas,asp,label='asp',color='b')
# plt.plot(particulas,trans,label='trans')
plt.plot(particulas,samp,label='samp',color='r')
plt.plot(particulas,sed,label='sed',color='y')
plt.plot(particulas,turb,label='turb',color='g')
plt.plot(particulas,inlet,label='inlet',color='k')
plt.plot(particulas,diff,label='diff',color='cyan',alpha=0.5)
# plt.xscale('log')


plt.plot(df_asp['x'],df_asp['Curve1'],'r',label='asp - PLC',linestyle='--')
plt.plot(df_diff['x'],1-df_diff['Curve1'],'cyan',label='diff - PLC',linestyle='--')
plt.plot(df_sedim['x'],1-df_sedim['Curve1'],'y',label='sedim - PLC',linestyle='--')
plt.plot(df_sampling['x'],df_sampling['Curve1'],'k',label='sampling - PLC',linestyle='--')

plt.legend()
plt.show()