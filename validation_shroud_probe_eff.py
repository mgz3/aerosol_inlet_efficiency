import matplotlib.pyplot as plt
from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp, pi, cos, sqrt, asin
from scipy.optimize import curve_fit
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA MUESTREO
# aspiracion
def n_asp(u0_u,stokes):
    n_aspiracion =  1 + (u0_u-1)*(1-(1+stokes*(2+0.617*(1/u0_u)))**-1) #valido para 0.005 < stokes < 2.03
    # n_aspiracion =  1 + (u0_u-1)*(1-(1+3.77*stokes**(0.883))**-1) #valido para 0.005 < stokes < 10
    return n_aspiracion


# transmision
def n_trans(u_u0,stokes):
    u0_u = 1/u_u0
    n_transport_inert = (1 + (u0_u - 1) / (1 + 2.66 / stokes ** (2 / 3))) / (1 + (u0_u - 1) / (1 + 0.418 / stokes))
    return n_transport_inert

# factor correccion Gong et al.
def factor_corr(u_u0,stokes,re):
    u0_u = 1/u_u0
    if not 8900 <= re <= 54000:
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
    # dp = dp/ 10 ** -6
    dp = dp* 10 ** -6
    cc = 1 + 1*(15.60+7*np.exp(-0.059*P*dp))/(P*dp)
    return cc


# .............................................
# sedimentacion
def n_sedim(Z):
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
    # Vts = tau_rel(rho_p,dp,mu,P,T)*9.81
    # print(Vts)
    eficiencia = np.exp(-4*Z/pi)
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
    # print(B)
    D = k_boltzman*T*B
    # D = k_boltzman*T/(3*pi*dp*mu)
    zita = pi *D*L/Q
    Sc=mu/(rho_f*D)
    Sh = 0.0118*re**(7/8)*Sc**(1/3)
    n = np.exp(-zita * Sh)
    return n

def n_turb_inert(d,L,Q,re,V,dp,rho_part,mu):
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
    stokes = core.Stokes_number(V, dp, d, rho_part, mu)
    print(dp, stokes)
    tau_mas = 0.0395*stokes*re**(3/4)
    if tau_mas > 12.9:
        V_mas = 0.1
    else:
        V_mas = 6*10**-4*tau_mas**2+2*10**-8*re
    Vt = V_mas*V/(5.03*re**(1/8))
    return np.exp(-(pi*d*L*Vt)/(Q))


def plot_eff(ax,diametros,eficiencias,color,label,grado=4,turb=False):
    ax.plot(diametros, eficiencias,color,label=label,marker='o',linestyle='',markersize=2)
    ax.legend()
    if turb:
        fit_exp_curve_exp(ax, diametros, eficiencias, color)
    else:
        fit_exp_curve(ax,diametros,eficiencias,color,grado)


def fit_exp_curve(ax,dp,n,color,grado=4):
    z = np.polyfit(dp, n, grado)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(dp[0], dp[-1], 50)
    y_new = f(x_new)
    ax.plot(x_new, y_new,color=color)

def func(x, a, b,c,d ):
    return c+a * np.exp(-b * x**2)
def fit_exp_curve_exp(ax,t,k,color):
    T_fit = t
    x = np.linspace(T_fit[0],T_fit[-1],50)
    yhat = np.array(k, dtype=float)
    popt_yhat, pcov_yhat = curve_fit(func, T_fit, yhat,maxfev=100000)
    ax.plot(x, func(x, *popt_yhat), label="",color=color)
# ----------------------------------------------------------------------------------------------------------------------
def replace_comma_with_dot(x):
    try:
        return float(x.replace(",", "."))
    except:
        return x
# # ------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------
# VALIDACION NUMERO DE STOKES
diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2,50])
diametros_micro = diametros_micro*10**-6
# DIMENSIONES [m]
d_shroud = 20*1e-3
d_inlet = 6*1e-3
d_inlet_out = 8*1e-3
L_offset= 30*1e-3
L_shroud = 80*1e-3
L_inlet=230*1e-3
# para hacer el analisis dividiendo en tramos
DOS_TRAMOS = False
L_inlet_1=15*1e-3 #largo del tramo 1
L_inlet_2=L_inlet-L_inlet_1 #largo del tramo 2
d_inlet_1= 6*1e-3
d_inlet_2=8*1e-3 #diametro del tramo 2
# PARAMETROS FLUIDO
level = 100 # altura en metros
atmos = ATMOSPHERE_1976(level)
U0 = 25 #m/s
U_U0 = 0.4
U_shroud = U0 * U_U0
Q_shroud =U_shroud*(pi*(d_shroud)**2/4)     #m**3/s  caudal volumetrico
Re_shroud = core.Reynolds(U0,d_shroud,atmos.rho,atmos.mu)
Q_inlet = 2    #L/min  caudal volumetrico estipulado por Renard
Q_inlet = Q_inlet/60000    #m**3/s
U_inlet = 4*Q_inlet/(d_inlet**2*pi)
Uinlet_Ushroud = U_inlet/U_shroud
Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)
# para hacer el analisis dividiendo en tramos
Re_inlet_1 = core.Reynolds(U_inlet,d_inlet_1,atmos.rho,atmos.mu)
Re_inlet_2 = core.Reynolds(U_inlet,d_inlet_2,atmos.rho,atmos.mu)
# PARTICULAS
rho_p = 1000 #kg/m**3    SUPOSICION  ESTANDAR
# for dp in diametros_micro:
#     stokes_shroud = core.Stokes_number(U_shroud, dp, d_shroud, rho_p, atmos.mu)
#     stokes_inlet = core.Stokes_number(U_inlet, dp, d_inlet, rho_p, atmos.mu)
#     print('-----------------------------------------------')
#     print('dp = {}'.format(dp))
#     print('Shroud stokes = {}'.format(stokes_shroud))
#     print('Inlet stokes = {}'.format(stokes_inlet))
# # ------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------
wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/'
# VALIDACION EF. ASPIRACION
file='asp_belyaev.csv'
df =pd.read_csv(wd+file)
df = df.applymap(replace_comma_with_dot)
# df['x']=np.exp(df['x'])

u0_u_2=[]
u0_u_4=[]
x = np.linspace(0.01,10,100000)
for stk in x:
    u0_u_2.append(n_asp(2,stk))
    u0_u_4.append(n_asp(4,stk))

plt.plot(x,u0_u_2,label='U0_U_2 calc')
plt.plot(x,u0_u_4,label='U0_U_4 calc')

plt.plot(df['x'],df['U0_U_2'],'r',label='U0_U_2')
plt.plot(df['x'],df['U0_U_4'],'b',label='U0_U_4')
plt.xscale('log')
plt.legend()

def calc_error_percent(curve1, curve2):
    # Obtenemos los valores x para ambas curvas
    x1 = np.arange(len(curve1))
    x2 = np.arange(len(curve2))

    # Interpolamos la curva con el tamaño más pequeño
    if len(curve1) < len(curve2):
        curve1 = np.interp(x2, x1, curve1)
    else:
        curve2 = np.interp(x1, x2, curve2)

    # error = 0
    # total = 0
    # for i in range(len(curve1)):
    #     error += abs(curve1[i] - curve2[i])
    #     total += curve1[i]
    # return (error / total) * 100
    error_porcentual = (np.abs(curve1 - curve2) / curve1) * 100
    return error_porcentual
error_percent = calc_error_percent(u0_u_4, df['U0_U_4'])
print("Error porcentual de u0_u=4 es:", np.mean(error_percent), "%")

error_percent = calc_error_percent(u0_u_2, df['U0_U_2'])
print("Error porcentual de u0_u=2 es:", np.mean(error_percent), "%")

plt.ylim(0,5)
plt.show()
# # ------------------------------------------------------------------------------------------------------------
# wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/'
# # VALIDACION EF. TRANSMISION
# file='transm_Belayev.csv'
# df =pd.read_csv(wd+file)
# df = df.applymap(replace_comma_with_dot)
# # df['x']=np.exp(df['x'])
#
# u0_u_2=[]
# u0_u_4=[]
# x = np.linspace(0.01,10,100000)
# for stk in x:
#     u0_u_2.append(n_trans(1/2,stk))
#     u0_u_4.append(n_trans(1/4,stk))
#
# plt.plot(x,u0_u_2,label='U0_U_2 calc')
# plt.plot(x,u0_u_4,label='U0_U_4 calc')
#
# plt.plot(df['x'],df['U0_U_2'],'r',label='U0_U_2')
# plt.plot(df['x'],df['U0_U_4'],'b',label='U0_U_4')
# plt.xscale('log')
# plt.legend()
#
# def calc_error_percent(curve1, curve2):
#     # Obtenemos los valores x para ambas curvas
#     x1 = np.arange(len(curve1))
#     x2 = np.arange(len(curve2))
#
#     # Interpolamos la curva con el tamaño más pequeño
#     if len(curve1) < len(curve2):
#         curve1 = np.interp(x2, x1, curve1)
#     else:
#         curve2 = np.interp(x1, x2, curve2)
#
#     error = 0
#     total = 0
#     for i in range(len(curve1)):
#         error += abs(curve1[i] - curve2[i])
#         total += curve1[i]
#     return (error / total) * 100
#
# error_percent = calc_error_percent(u0_u_4, df['U0_U_4'])
# print("Error porcentual de u0_u=4 es:", error_percent, "%")
#
# error_percent = calc_error_percent(u0_u_2, df['U0_U_2'])
# print("Error porcentual de u0_u=2 es:", error_percent, "%")
#
# plt.show()
# ------------------------------------------------------------------------------------------------------------
# # VALIDACION EF. MUESTREO
# wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/'
# file='inlet_belayev.csv'
# df =pd.read_csv(wd+file)
# df = df.applymap(replace_comma_with_dot)
# # df['x']=np.exp(df['x'])
#
# u0_u_2=[]
# u0_u_4=[]
# x = np.linspace(0.01,10,100000)
# for stk in x:
#     u0_u_2.append(n_asp(2,stk)*n_trans(1/2,stk))
#     u0_u_4.append(n_asp(4,stk)*n_trans(1/4,stk))
#
# plt.plot(x,u0_u_2,label='U0_U_2 calc')
# plt.plot(x,u0_u_4,label='U0_U_4 calc')
#
# plt.plot(df['x'],df['U0_U_2'],'r',label='U0_U_2')
# plt.plot(df['x'],df['U0_U_4'],'b',label='U0_U_4')
# plt.xscale('log')
# plt.legend()
#
# def calc_error_percent(curve1, curve2):
#     # Obtenemos los valores x para ambas curvas
#     x1 = np.arange(len(curve1))
#     x2 = np.arange(len(curve2))
#
#     # Interpolamos la curva con el tamaño más pequeño
#     if len(curve1) < len(curve2):
#         curve1 = np.interp(x2, x1, curve1)
#     else:
#         curve2 = np.interp(x1, x2, curve2)
#
#     error = 0
#     total = 0
#     for i in range(len(curve1)):
#         error += abs(curve1[i] - curve2[i])
#         total += curve1[i]
#     return (error / total) * 100
#
# error_percent = calc_error_percent(u0_u_4, df['U0_U_4'])
# print("Error porcentual de u0_u=4 es:", error_percent, "%")
#
# error_percent = calc_error_percent(u0_u_2, df['U0_U_2'])
# print("Error porcentual de u0_u=2 es:", error_percent, "%")
# plt.show()

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# # VALIDACION EF. SEDIMENTACION
# # ---------------------------------------------------------------------------------------------------------
# wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/'
# file='grav_eff.csv'
# df =pd.read_csv(wd+file)
# df = df.applymap(replace_comma_with_dot)
# # df['x']=np.exp(df['x'])
#
# u0_u_=[]
# z = []
# # diametros_micro = np.linspace(0.001,10,100000)
# for dp in diametros_micro:
#     Vts = tau_rel(rho_p, dp, atmos.mu, atmos.P, atmos.T) * 9.81
#     Z = pi/4*d_inlet*L_inlet*Vts/Q_inlet
#     z.append(Z)
#     u0_u_.append(n_sedim(Z))
#
# plt.plot(z,u0_u_,label='sedim calc')
#
# plt.plot(df['x'],df['U0_U_'],'r',label='sedim')
# plt.xscale('log')
# plt.legend()
# plt.show()
# ---------------------------------------------------------------------------------------------------------
# VALIDACION EF. SEDIMENTACION
# ---------------------------------------------------------------------------------------------------------
# wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/'
# file='diff_eff.csv'
# df =pd.read_csv(wd+file)
# df = df.applymap(replace_comma_with_dot)
# # df['x']=np.exp(df['x'])
#
# u0_u_=[]
# d = 0.0127
# L = 2
# re = 3000
# v = re*atmos.mu/(atmos.rho*L)
# q = v*d**2*pi/4
# factor = 1*10**-6
# diametros_micro = np.linspace(10**-4*factor,10**-1*factor,100000)
# for dp in diametros_micro:
#     # print(v)
#     u0_u_.append(n_dif(dp,atmos.P,atmos.T,2,q,atmos.mu,re,atmos.rho))
#
# x_diametro_interes = [0.2,0.2]
# y_diametro_interes = [0,1]
# plt.plot(x_diametro_interes,y_diametro_interes,label='limite inferio LOAC')
#
# plt.plot(diametros_micro,u0_u_,label='DIFF calc')
# plt.plot(df['x'],df['U0_U_'],'r',label='DIFF')
# plt.xscale('log')
# plt.legend()
# plt.show()



# def Cc(dp, P, T):
#     P = P * 10 ** -3
#     # dp = dp/ 10 ** -6
#     dp = dp* 10 ** -6
#     cc = 1 + 1*(15.60+7*np.exp(-0.059*P*dp))/(P*dp)
#     return cc
# from fluids import core, ATMOSPHERE_1976
# level = 100
# atmos = ATMOSPHERE_1976(level)
# diametros_micro = np.linspace(0.0001,50,10000)
# CC = []
# for i in diametros_micro:
#     CC.append(Cc())

# ------------------------------------------------------------------------------------------------------------
# VALIDACION EF. SEDIMENTACION
# ---------------------------------------------------------------------------------------------------------
# wd = '/home/mgonzalez/Documentos/Digitalizacion_Aerosoles_Eff/'
# file='grav_eff.csv'
# df =pd.read_csv(wd+file)
# df = df.applymap(replace_comma_with_dot)
# df['x']=np.exp(df['x'])

# u0_u_=[]
# z = []
# diametros_micro = np.linspace(0.00000000000000000001,0.1,100000)
# for dp in diametros_micro:
#     Vts = tau_rel(rho_p, dp, atmos.mu, atmos.P, atmos.T) * 9.81
#     Z = pi/4*d_inlet*L_inlet*Vts/Q_inlet
#     z.append(Z)
#     u0_u_.append(n_sedim(Z))
#
# Z = np.linspace(0.001,10,1000)
# u0_u_=[]
# z =[]
# for zs in Z:
#     z.append(zs)
#     u0_u_.append(n_sedim(zs))
# plt.plot(z,u0_u_,label='sedim calc')
#
# plt.plot(df['x'],df['U0_U_'],'r',label='sedim')
# plt.xscale('log')
# plt.legend()
# plt.show()
# # ---------------------------------------------------------------------------------------------------------
