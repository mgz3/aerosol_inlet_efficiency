def n_iner_codo(V, dp, d, rho_p,ang):
    """

    Args:
        Stks: numero de stokes
        ang: angulo de curvatura en grados

    Returns: rendimiento inercial debido a un codo

    """
    # thetha = ang*pi/180
    # delta= 10/4
    # a=-0.9526-0.05686*delta
    # b=(-0.297-0.0174*delta)/(1-0.07*delta+0.0171*delta**2)
    # c=-0.306+1.895/sqrt(delta)-2./delta
    # d = (0.131-0.132*delta+0.000383*delta**2)/(1-0.129*delta+0.0136*delta**2)
    # efficiency_bend=exp((4.61+a*thetha*Stks)/(1+b*thetha*Stks+c*thetha*Stks**2+d*thetha**2*Stks))

    # print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    # print('valores del codito')
    # if Stks > 1:
    #     print('ACAAAAAAA ESTUPIDOOOOO   ---------------------------------------------------1231234-2341234-13-423-4-23')
    # print('stokes : ',Stks)
    # print('ang en grados : {}    ang en radianes: {}'.format(ang,ang*pi/180))
    # print('-.-.-.--.-.-.-.-.-.-.-.-.-.-.')
    Stks = core.Stokes_number(V, dp, d, rho_p, atmos.mu)
    return exp(-2.823*Stks*ang*pi/180)
    # return efficiency_bend


from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp, pi, cos, sqrt, asin
from scipy.optimize import curve_fit

# tramo I
d_inlet = 8*10**-3                      # mm diametro interno de entrada de la sonda
LI_int= 30*10**-3                              # mm long tramo I                             # m/s velocidad en diam_1
ang_rot = 60                            #ang de rotacion en grados
# tramo II
LII_int= 40*10**-3                              # mm long tramo II
# PARAMETROS FLUIDO
level = 100 # altura en metros
atmos = ATMOSPHERE_1976(level)
U0 = 20 #m/s
Q_inlet = 2    #L/min  caudal volumetrico estipulado por Renard
Q_inlet = Q_inlet/60000    #m**3/s
U_inlet = 4*Q_inlet/(d_inlet**2*pi)
U0_Uinlet = U0/U_inlet 
Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)
Re_inlet1=Re_inlet
DOS_TRAMOS = True
L_inlet_1=30*10**-3  #largo del tramo 1
L_inlet_2=LII_int#largo del tramo 2
d_inlet_1= 8*1e-3
d_inlet_2=8*1e-3 #diametro del tramo 2

# PARTICULAS
rho_p = 1000 #kg/m**3    SUPOSICION  ESTANDAR
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
inlet_elbow=[]
EFICIENCIAS_TOTAL = []
FACTOR = []
print('SHROUD')

print('-----------------------------------')
print('INLET')
for dp in diametros_micro:
    # ..............................................
    # INLET
    if DOS_TRAMOS:
        eff_aspiracion = n_asp(U0_Uinlet, dp, U_inlet, d_inlet)
        eff_transmision =n_trans(U0_Uinlet, dp, U_inlet, d_inlet)
        eff_diffusion = n_dif(dp, atmos.P, atmos.T, L_inlet_1, Q_inlet, atmos.mu, Re_inlet1, atmos.rho)*n_dif(dp, atmos.P, atmos.T, L_inlet_2, Q_inlet, atmos.mu, Re_inlet1, atmos.rho)
        eff_sedimentacion = n_sedim(d_inlet_1, L_inlet_1, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q_inlet, U_inlet, Re_inlet1)*n_sedim(d_inlet_2, L_inlet_2, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q_inlet, U_inlet, Re_inlet)
        eff_turbulen = n_turb_inert(d_inlet_1, L_inlet_1, Q_inlet, Re_inlet, U_inlet, dp)*n_turb_inert(d_inlet_2, L_inlet_2, Q_inlet, Re_inlet, U_inlet, dp)
        eff_elbow = n_iner_codo(U_inlet,dp,d_inlet_1,rho_p,ang_rot)

        inlet_elbow.append(eff_elbow)
        inlet_eff_asp.append(eff_aspiracion)
        inlet_eff_transm.append(eff_transmision)
        inlet_eff_diff.append(eff_diffusion)
        inlet_eff_sed.append(eff_sedimentacion)
        inlet_eff_turb.append(eff_turbulen)
    else:
        inlet_eff_asp.append(n_asp(U0_Uinlet,dp,U_inlet,d_inlet))
        inlet_eff_transm.append(n_trans(U0_Uinlet,dp,U_inlet,d_inlet))
        inlet_eff_diff.append(n_dif(dp,atmos.P,atmos.T,L_inlet_1,Q_inlet,atmos.mu,Re_inlet,atmos.rho))
        inlet_eff_sed.append(n_sedim(d_inlet,L_inlet_1,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_inlet,U_inlet,Re_inlet))
        inlet_eff_turb.append(n_turb_inert(d_inlet,L_inlet_1,Q_inlet,Re_inlet,U_inlet,dp))





for i in range(len(diametros_micro)):
    # INLET
    EFICIENCIAS_INLET_ASP.append(inlet_eff_asp[i]*inlet_eff_transm[i])
    EFICIENCIAS_INLET_TRANSP.append(inlet_eff_sed[i]*inlet_eff_diff[i]*inlet_eff_turb[i]*inlet_elbow[i])
    EFICIENCIAS_INLET.append(EFICIENCIAS_INLET_ASP[i]*EFICIENCIAS_INLET_TRANSP[i])

import matplotlib.pyplot as plt

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


fig1, ax = plt.subplots(1,1)
plt.style.use(['science','ieee'])
ax.plot([0,50],[1,1],linestyle='dashed',color='gray')
eff_inlet = ax.plot(diametros_micro, EFICIENCIAS_INLET,label='Sonda de muestreo',color='k',marker='o',linestyle='',markersize=2)
fit_exp_curve(ax,diametros_micro,EFICIENCIAS_INLET,color='k')
maximo=np.max(EFICIENCIAS_INLET)
ax.legend(fontsize=30,markerscale=2,framealpha=1,loc='best')
ax.grid()
ax.set_xlabel(r'Diametro Particula, dp [$\mu$m]')
ax.set_ylabel(r'Eficiencia $\eta$')

plot_bands = False
band_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.1,3,5,7.5,10,12.5,15,17.5,20,22,30,40,50]
banda_fill = []
if plot_bands:
    from matplotlib.legend import Legend
    for i in range(len(band_list)-1):
        rango=[band_list[i],band_list[i+1]]
        fill = ax.fill_between(rango,0,maximo,alpha=0.5)
        banda_fill.append(fill)
    legend2=Legend(ax,banda_fill,['{} - {}'.format(band_list[i],band_list[i+1]) for i in range(len(band_list)-1)],
          loc=1,labelcolor='linecolor',title=r'LOAC Ranges [$\mu$m]',framealpha=1,
                   prop={'weight':'black','size':11})
    ax.add_artist(legend2)

plt.rcParams['text.usetex'] = True
ax.autoscale(tight=True)
# ax.set_xlabel(r'Particle Diameter, dp ($\mu$m)')
# ax.set_ylabel(r'Efficiency $\eta$')
# ax.legend(loc='best',bbox_to_anchor=(0.3, 0.3, 0.5, 0.5))#,bbox_to_anchor=(-0.1, -0.1))
# ax.legend(fontsize='4')
ax.grid(True)
plt.show()
print('done')
