from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp, pi, cos, sqrt, asin
from scipy.optimize import curve_fit

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
level = 0 # altura en metros
atmos = ATMOSPHERE_1976(level)
U0 = 25 #m/s
if U0 >= 8 and U0 < 11.5:
    U_U0 = 0.3
elif U0 >= 11.5 and U0 < 14:
    U_U0 = 0.25
elif U0 >= 14 and U0 < 25:
    U_U0 = 0.2
else:
    U_U0 = 0.22

print('VALOR DE U/U0: {}'.format(U_U0))
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
diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2,50])
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
    # Vt = (6*(10**-4)*(0.0395*stokes*re**(3/4))**2 + 2*(10**-8)*re)*V
    return np.exp(-(pi*d*L*Vt)/(Q))
# ----------------------------------------------------------------------------------------------------------------------
EFICIENCIAS_SHROUD = []
EFICIENCIAS_SHROUD_ASP = []
EFICIENCIAS_SHROUD_TRANSP = []
shroud_eff_asp = []
shroud_eff_transm = []
shroud_eff_sed = []
shroud_eff_diff = []
shroud_eff_turb = []
EFICIENCIAS_INLET = []
EFICIENCIAS_INLET_ASP = []
EFICIENCIAS_INLET_TRANSP = []
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
    if DOS_TRAMOS:
        eff_aspiracion = n_asp(Uinlet_Ushroud, dp, U_inlet, d_inlet)
        eff_transmision =n_trans(Uinlet_Ushroud, dp, U_inlet, d_inlet)
        eff_diffusion = n_dif(dp, atmos.P, atmos.T, L_inlet_1, Q_inlet, atmos.mu, Re_inlet_1, atmos.rho)*n_dif(dp, atmos.P, atmos.T, L_inlet_2, Q_inlet, atmos.mu, Re_inlet_2, atmos.rho)
        eff_sedimentacion = n_sedim(d_inlet_1, L_inlet_1, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q_inlet, U_inlet, Re_inlet_1)*n_sedim(d_inlet_2, L_inlet_2, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q_inlet, U_inlet, Re_inlet_2)
        eff_turbulen = n_turb_inert(d_inlet_1, L_inlet_1, Q_inlet, Re_inlet, U_inlet, dp)*n_turb_inert(d_inlet_2, L_inlet_2, Q_inlet, Re_inlet, U_inlet, dp)

        inlet_eff_asp.append(eff_aspiracion)
        inlet_eff_transm.append(eff_transmision)
        inlet_eff_diff.append(eff_diffusion)
        inlet_eff_sed.append(eff_sedimentacion)
        inlet_eff_turb.append(eff_turbulen)
    else:
        inlet_eff_asp.append(n_asp(Uinlet_Ushroud,dp,U_inlet,d_inlet))
        inlet_eff_transm.append(n_trans(Uinlet_Ushroud,dp,U_inlet,d_inlet))
        inlet_eff_diff.append(n_dif(dp,atmos.P,atmos.T,L_inlet,Q_inlet,atmos.mu,Re_inlet,atmos.rho))
        inlet_eff_sed.append(n_sedim(d_inlet,L_inlet,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_inlet,U_inlet,Re_inlet))
        inlet_eff_turb.append(n_turb_inert(d_inlet,L_inlet,Q_inlet,Re_inlet,U_inlet,dp))
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------




for i in range(len(diametros_micro)):
    # SHROUD
    EFICIENCIAS_SHROUD_ASP.append(shroud_eff_asp[i]*shroud_eff_transm[i])
    EFICIENCIAS_SHROUD_TRANSP.append(shroud_eff_sed[i]*shroud_eff_diff[i]*shroud_eff_turb[i])
    # EFICIENCIAS_SHROUD.append(shroud_eff_asp[i]*shroud_eff_transm[i]*shroud_eff_sed[i]*shroud_eff_diff[i]*shroud_eff_turb[i])
    EFICIENCIAS_SHROUD.append(EFICIENCIAS_SHROUD_ASP[i]*EFICIENCIAS_SHROUD_TRANSP[i])

    # INLET
    EFICIENCIAS_INLET_ASP.append(inlet_eff_asp[i]*inlet_eff_transm[i])
    EFICIENCIAS_INLET_TRANSP.append(inlet_eff_sed[i]*inlet_eff_diff[i]*inlet_eff_turb[i])
    # EFICIENCIAS_INLET.append(inlet_eff_asp[i]*inlet_eff_transm[i]*inlet_eff_sed[i]*inlet_eff_diff[i]*inlet_eff_turb[i])
    EFICIENCIAS_INLET.append(EFICIENCIAS_INLET_ASP[i]*EFICIENCIAS_INLET_TRANSP[i])
    # TOTAL
    # EFICIENCIAS_TOTAL.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i]*FACTOR[i])
    EFICIENCIAS_TOTAL.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i])
    maximo = np.max([EFICIENCIAS_TOTAL[i],EFICIENCIAS_SHROUD[i],EFICIENCIAS_INLET[i]])

import matplotlib.pyplot as plt
fig1, axs = plt.subplots(3,2,constrained_layout = True)
for ax in axs.flat:
    ax.grid()
    ax.set(xlabel=r'Diametro $\mu$m', ylabel=r'Eficiencia $\eta$')


ax1 = axs[0, 0] # sampling shroud
ax1.set_title('Muestreo Envoltura')
ax2 = axs[0, 1] # sampling inlet
ax2.set_title('Muestreo Toma')
ax3 = axs[1, 0] # transport shroud
ax3.set_title('Transporte Envoltura')
ax4 = axs[1, 1] # transport inlet
ax4.set_title('Transporte Toma')
ax5 = axs[2, 0] # total shroud
ax5.set_title('Total Envoltura')
ax6 = axs[2, 1] # total inlet
ax6.set_title('Total Toma')


diametros_micro = diametros_micro/10**-6

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


# SHROUD
plot_eff(ax1,diametros_micro,shroud_eff_asp,'b','Asp. Envoltura')
plot_eff(ax1,diametros_micro,shroud_eff_transm,'r','Transm. Envoltura')
plot_eff(ax3,diametros_micro,shroud_eff_diff,'r','Difusion')
plot_eff(ax3,diametros_micro,shroud_eff_turb,'g','Turbulento',turb=True)
plot_eff(ax3,diametros_micro,shroud_eff_sed,'b','Sedimentacion')
plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD,'k','TOTAL ENVOLTURA')
plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD_ASP,'cyan','ENVOLTURA ASPIRACION')
plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD_TRANSP,'m','ENVOLTURA TRANSPORTE')
# INLET
plot_eff(ax2,diametros_micro,inlet_eff_asp,'b','Asp. Toma')
plot_eff(ax2,diametros_micro,inlet_eff_transm,'r','Transm. Toma')
plot_eff(ax4,diametros_micro,inlet_eff_diff,'r','Difusion')
plot_eff(ax4,diametros_micro,inlet_eff_turb,'g','Turbulento',turb=True)
plot_eff(ax4,diametros_micro,inlet_eff_sed,'b','Sedimentacion')
plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET,'k','TOTAL TOMA')
plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET_ASP,'cyan','TOMA ASPIRACION')
plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET_TRANSP,'m','TOMA TRANSPORTE')


fig2, ax = plt.subplots(1,1)

ax.plot([0,50],[1,1],linestyle='dashed',color='gray')
eff_total = ax.plot(diametros_micro, EFICIENCIAS_TOTAL,label='EFICIENCIA TOTAL',color='k',marker='o',linestyle='',markersize=2)
fit_exp_curve(ax,diametros_micro,EFICIENCIAS_TOTAL,color='k')
eff_inlet = ax.plot(diametros_micro, EFICIENCIAS_INLET,label='EFICIENCIA TOTAL TOMA',color='cyan',marker='o',linestyle='',markersize=2)
fit_exp_curve(ax,diametros_micro,EFICIENCIAS_INLET,color='cyan')
eff_shroud = ax.plot(diametros_micro, EFICIENCIAS_SHROUD,label='EFICIENCIA TOTAL ENVOLTURA',color='m',marker='o',linestyle='',markersize=2)
fit_exp_curve(ax,diametros_micro,EFICIENCIAS_SHROUD,color='m')
# eff_factor = ax.plot(diametros_micro, FACTOR,label='FACTOR',color='dimgray',marker='o',linestyle='',markersize=2)
# fit_exp_curve(ax,diametros_micro,FACTOR,color='dimgray')
ax.set_xlim(0,55)
ax.set_ylim(0,maximo)

ax.grid()
# ax.set_xlabel(r'Particle Diameter, dp [$\mu$m]')
ax.set_xlabel(r'Diametro de Particula, dp [$\mu$m]')
# ax.set_ylabel(r'Efficiency $\eta$')
ax.set_ylabel(r'Eficiencia $\eta$')

plot_bands = True
band_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.1,3,5,7.5,10,12.5,15,17.5,20,22,30,40,50]
banda_fill = []
if plot_bands:
    from matplotlib.legend import Legend
    for i in range(len(band_list)-1):
        rango=[band_list[i],band_list[i+1]]
        fill = ax.fill_between(rango,0,maximo,alpha=0.5)
        banda_fill.append(fill)
    legend2=Legend(ax,banda_fill,['{} - {}'.format(band_list[i],band_list[i+1]) for i in range(len(band_list)-1)],
          loc=1,labelcolor='linecolor',title=r'LOAC Rangos [$\mu$m]',framealpha=1,
                   prop={'weight':'black','size':11})
    ax.add_artist(legend2)

# plot de eficiencias de JUAN SALDIA
import pandas as pd
def replace_comma_with_dot(x):
    try:
        return float(x.replace(",", "."))
    except:
        return x
FOLDER = '/home/maxpower/PycharmProjects/aerosol_inlet_efficiency/CFD_JUAN/'
file = 'eff_{}ms.csv'.format(str(int(U0)))
df = pd.read_csv(FOLDER + file)
df = df.applymap(replace_comma_with_dot)
plt.plot(df['x'],df['Curve1'],'crimson',label='SALDIA {}'.format(str(int(U0)) + ' m/s'),linestyle='--')
ax.legend(fontsize=15,markerscale=2,framealpha=1,loc=2)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
fig3, ax = plt.subplots(1,1)

FOLDER = '/home/maxpower/PycharmProjects/aerosol_inlet_efficiency/CFD_JUAN/'
file = 'asp_{}.csv'.format(str(int(U0)))
df = pd.read_csv(FOLDER + file)
df = df.applymap(replace_comma_with_dot)
plt.plot(df['x'],df['Curve1'],'y',label='SALDIA {}'.format(str(int(U0)) + ' m/s'),linestyle='--')

plot_eff(ax,diametros_micro,shroud_eff_asp,'b','Asp. Envoltura')


ax.legend(fontsize=15,markerscale=2,framealpha=1,loc=2)
ax.grid()
ax.set_xlabel(r'Diametro de Particula, dp [$\mu$m]')
ax.set_ylabel(r'Eficiencia $\eta$')

plt.show()
print('done')
