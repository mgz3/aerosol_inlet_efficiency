from fluids import core, ATMOSPHERE_1976
import numpy as np
from math import exp, pi, cos, sqrt, asin
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# VALORES PARAMETROS SIPPOLA ET AL.
#
x_asp_5ms_sipp = [
0.2589,
0.7241,
1.2742,
2.2208,
3.702,
4.6952,
6.158,
8.543,
9.7544,
10.9278,
11.8364,
12.1392,
13.3505,
14.5619,
15.7353,
16.606,
16.9467,
18.158,
18.9813,
19.7194,
19.956
]
asp_5ms_sipp = [
1.00418,
0.99854,
1.00149,
1.00149,
1.04515,
1.08457,
1.13582,
1.26119,
1.32388,
1.40448,
1.45821,
1.47612,
1.55672,
1.63731,
1.72687,
1.7806,
1.80746,
1.89701,
1.94851,
2.00896,
2.01791
]
x_trans_5ms_sipp = [
0.1326,
1.0942,
1.2866,
2.2482,
3.1713,
4.133,
5.1715,
6.0947,
7.0563,
7.9794,
8.0564,
8.8256,
9.7873,
10.7104,
11.6335,
12.6336,
13.4798,
14.4415,
15.3646,
16.3262,
17.2494,
18.2111,
19.1727
]
transm_5ms_sipp = [
1.00172,
1.00146,
1.00141,
1.00115,
1.03718,
1.04599,
1.07292,
1.09081,
1.10869,
1.14471,
1.14469,
1.16262,
1.18957,
1.21653,
1.24349,
1.27042,
1.28834,
1.30621,
1.3241,
1.33291,
1.3508,
1.35055,
1.35936
]
# VALORES PARAMETROS McFARLAND ET AL.
#
x_asp_farland = [
1.1821,
2.1077,
2.6866,
3.574,
4.5,
5.1173,
6.0045,
6.9304,
7.8177,
8.7434,
9.6307,
9.7854,
10.6335,
11.5207,
12.4462,
13.3719,
14.2973,
]
asp_farland = [
1.0158,
1.03464,
1.05742,
1.08392,
1.11424,
1.13318,
1.15586,
1.18235,
1.20885,
1.23152,
1.25803,
1.26945,
1.28068,
1.30336,
1.3222,
1.34487,
1.35989
]
x_transm_farland = [
0.1259,
0.5931,
0.9027,
1.3097,
1.7169,
2.0569,
2.476,
3.0468,
3.7271,
4.6524,
5.1148,
6.04,
7.1575,
8.121,
8.7762,
9.7394,
10.6256,
11.5505,
12.1284,
13.0916,
13.8622,
14.7867,
]
transm_farland = [
0.99966,
0.99952,
1.00244,
1.01316,
1.02629,
1.03161,
1.03992,
1.05059,
1.06093,
1.07595,
1.07582,
1.08702,
1.09051,
1.09787,
1.1015,
1.10122,
1.10095,
1.1045,
1.10433,
1.10405,
1.10382,
1.09972,
]

# DIMENSIONES [m]
d_shroud = 28*1e-3
d_inlet = 8*1e-3
d_inlet_out = 17*1e-3
L_offset= 45*1e-3
L_shroud = 100*1e-3
L_inlet=75*1e-3
# para hacer el analisis dividiendo en tramos
DOS_TRAMOS = False
L_inlet_1=15*1e-3 #largo del tramo 1
L_inlet_2=L_inlet-L_inlet_1 #largo del tramo 2
d_inlet_1= 6*1e-3
d_inlet_2=8*1e-3 #diametro del tramo 2

# PARAMETROS FLUIDO
level = 100 # altura en metros
U0 = 8.8 #m/s
U_U0 = 0.4
Q_inlet = 5    #L/min  caudal volumetrico estipulado por Renard

atmos = ATMOSPHERE_1976(level)
U_shroud = U0 * U_U0
Q_shroud =U_shroud*(pi*(d_shroud)**2/4)     #m**3/s  caudal volumetrico
Re_shroud = core.Reynolds(U0,d_shroud,atmos.rho,atmos.mu)
Q_inlet = Q_inlet/60000    #m**3/s
U_inlet = 4*Q_inlet/(d_inlet**2*pi)
Uinlet_Ushroud = U_inlet/U_shroud
Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)
# para hacer el analisis dividiendo en tramos
Re_inlet_1 = core.Reynolds(U_inlet,d_inlet_1,atmos.rho,atmos.mu)
Re_inlet_2 = core.Reynolds(U_inlet,d_inlet_2,atmos.rho,atmos.mu)
# PARTICULAS
rho_p = 1000 #kg/m**3    SUPOSICION  ESTANDAR
diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2])
diametros_micro = diametros_micro*10**-6


# ----------------------------------------------------------------------------------------------------------------------
# # VALORES PARAMETROS SONDA TERO
# # DIMENSIONES [m]
# d_shroud = 20*1e-3
# d_inlet = 6*1e-3
# d_inlet_out = 8*1e-3
# L_offset= 30*1e-3
# L_shroud = 80*1e-3
# L_inlet=230*1e-3
# # para hacer el analisis dividiendo en tramos
# DOS_TRAMOS = False
# L_inlet_1=15*1e-3 #largo del tramo 1
# L_inlet_2=L_inlet-L_inlet_1 #largo del tramo 2
# d_inlet_1= 6*1e-3
# d_inlet_2=8*1e-3 #diametro del tramo 2
#
# # PARAMETROS FLUIDO
# level = 100 # altura en metros
# atmos = ATMOSPHERE_1976(level)
# U0 = 25 #m/s
# U_U0 = 0.4
# U_shroud = U0 * U_U0
# Q_shroud =U_shroud*(pi*(d_shroud)**2/4)     #m**3/s  caudal volumetrico
# Re_shroud = core.Reynolds(U0,d_shroud,atmos.rho,atmos.mu)
# Q_inlet = 2    #L/min  caudal volumetrico estipulado por Renard
# Q_inlet = Q_inlet/60000    #m**3/s
# U_inlet = 4*Q_inlet/(d_inlet**2*pi)
# Uinlet_Ushroud = U_inlet/U_shroud
# Re_inlet = core.Reynolds(U_inlet,d_inlet,atmos.rho,atmos.mu)
# # para hacer el analisis dividiendo en tramos
# Re_inlet_1 = core.Reynolds(U_inlet,d_inlet_1,atmos.rho,atmos.mu)
# Re_inlet_2 = core.Reynolds(U_inlet,d_inlet_2,atmos.rho,atmos.mu)
#
# # PARTICULAS
# rho_p = 1000 #kg/m**3    SUPOSICION  ESTANDAR
# diametros_micro = np.array([0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2,50])
# diametros_micro = diametros_micro*10**-6

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

    k = 3*L*tau_rel(rho_p,dp,mu,P,T)*9.81/(4*d*V)
    eficiencia_sippola =1-2/pi*(2*k*sqrt(1-k**(2/3))-k**(1/3)*sqrt(1-k**(2/3))+asin(k**(1/3)))
    return eficiencia_sippola


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
# ----------------------------------------------------------------------------------------------------------------------

ratios = np.linspace(0.1,1,10)
ratios= [0.4]
for ratio in ratios:
    # # DIMENSIONES [m]  Sippola
    # d_shroud = 28 * 1e-3
    # d_inlet = 8 * 1e-3
    # d_inlet_out = 17 * 1e-3
    # L_offset = 45 * 1e-3
    # L_shroud = 100 * 1e-3
    # L_inlet = 75 * 1e-3
    # # para hacer el analisis dividiendo en tramos
    # DOS_TRAMOS = True
    # L_inlet_1 = 50 * 1e-3  # largo del tramo 1
    # L_inlet_2 = L_inlet - L_inlet_1  # largo del tramo 2
    # d_inlet_1 = 8 * 1e-3
    # d_inlet_2 = 17 * 1e-3  # diametro del tramo 2
    #
    # # PARAMETROS FLUIDO
    # level = 0  # altura en metros
    # U0 = 5.3  # m/s
    # U_U0 = ratio
    # Q_inlet = 5  # L/min  caudal volumetrico

    # DIMENSIONES[m]  McFarland
    d_shroud = 102 * 1e-3
    d_inlet = 30 * 1e-3
    d_inlet_out = 51 * 1e-3
    L_offset = 153 * 1e-3
    L_shroud = 406 * 1e-3
    L_inlet = 253 * 1e-3
    # para hacer el analisis dividiendo en tramos
    DOS_TRAMOS = False
    L_inlet_1 = 50 * 1e-3  # largo del tramo 1
    L_inlet_2 = L_inlet - L_inlet_1  # largo del tramo 2
    d_inlet_1 = 8 * 1e-3
    d_inlet_2 = 17 * 1e-3  # diametro del tramo 2

    # PARAMETROS FLUIDO
    level = 0  # altura en metros
    U0 = 14  # m/s
    U_U0 = ratio
    Q_inlet = 170  # L/min  caudal volumetrico

    atmos = ATMOSPHERE_1976(level)
    U_shroud = U0 * U_U0
    print(U_shroud, ' u_shroud' , U_U0, 'u_u0')
    Q_shroud = U_shroud * (pi * (d_shroud) ** 2 / 4)  # m**3/s  caudal volumetrico
    Re_shroud = core.Reynolds(U0, d_shroud, atmos.rho, atmos.mu)
    Q_inlet = Q_inlet / 60000  # m**3/s
    U_inlet = 4 * Q_inlet / (d_inlet ** 2 * pi)
    Uinlet_Ushroud = U_inlet / U_shroud
    Re_inlet = core.Reynolds(U_inlet, d_inlet, atmos.rho, atmos.mu)
    # para hacer el analisis dividiendo en tramos
    Re_inlet_1 = core.Reynolds(U_inlet, d_inlet_1, atmos.rho, atmos.mu)
    Re_inlet_2 = core.Reynolds(U_inlet, d_inlet_2, atmos.rho, atmos.mu)
    # PARTICULAS
    rho_p = 1000  # kg/m**3    SUPOSICION  ESTANDAR
    diametros_micro = np.array(
        [0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1, (1.1 + 3) / 2, (3 + 5) / 2, (5 + 7.5) / 2, (7.5 + 10) / 2,
         (10 + 12.5) / 2, (12.5 + 15) / 2, (15 + 17.5) / 2, (17.5 + 20) / 2, (20 + 22) / 2])
    diametros_micro = diametros_micro * 10 ** -6
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



    ASPIRACION = []
    for i in range(len(diametros_micro)):
        # SHROUD
        EFICIENCIAS_SHROUD_ASP.append(shroud_eff_asp[i])
        # EFICIENCIAS_SHROUD_ASP.append(shroud_eff_asp[i]*shroud_eff_transm[i])
        EFICIENCIAS_SHROUD_TRANSP.append(shroud_eff_sed[i]*shroud_eff_diff[i]*shroud_eff_turb[i])
        # EFICIENCIAS_SHROUD.append(shroud_eff_asp[i]*shroud_eff_transm[i]*shroud_eff_sed[i]*shroud_eff_diff[i]*shroud_eff_turb[i])
        EFICIENCIAS_SHROUD.append(EFICIENCIAS_SHROUD_ASP[i]*EFICIENCIAS_SHROUD_TRANSP[i])

        # INLET
        # EFICIENCIAS_INLET_ASP.append(inlet_eff_asp[i]*inlet_eff_transm[i])
        EFICIENCIAS_INLET_ASP.append(inlet_eff_asp[i])
        EFICIENCIAS_INLET_TRANSP.append(inlet_eff_sed[i]*inlet_eff_diff[i]*inlet_eff_turb[i])
        # EFICIENCIAS_INLET.append(inlet_eff_asp[i]*inlet_eff_transm[i]*inlet_eff_sed[i]*inlet_eff_diff[i]*inlet_eff_turb[i])
        EFICIENCIAS_INLET.append(EFICIENCIAS_INLET_ASP[i]*EFICIENCIAS_INLET_TRANSP[i])
        # EFICIENCIAS_INLET.append(EFICIENCIAS_INLET_ASP[i]) #para validar
        # TOTAL
        EFICIENCIAS_TOTAL.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i]*FACTOR[i])
        ASPIRACION.append(EFICIENCIAS_INLET_ASP[i]*EFICIENCIAS_SHROUD_ASP[i])
        # EFICIENCIAS_TOTAL.append(EFICIENCIAS_SHROUD[i]*EFICIENCIAS_INLET[i]) #para validar con Sippola
        maximo = np.max([EFICIENCIAS_TOTAL[i],EFICIENCIAS_SHROUD[i],EFICIENCIAS_INLET[i]])

    diametros_micro = diametros_micro/10**-6

    # fig1, axs = plt.subplots(3,2,constrained_layout = True)
    # for ax in axs.flat:
    #     ax.grid()
    #     ax.set(xlabel=r'Diametro $\mu$m', ylabel=r'Eficiencia $\eta$')
    #
    # ax1 = axs[0, 0] # sampling shroud
    # ax1.set_title('sampling shroud')
    # ax2 = axs[0, 1] # sampling inlet
    # ax2.set_title('sampling inlet')
    # ax3 = axs[1, 0] # transport shroud
    # ax3.set_title('transport shroud')
    # ax4 = axs[1, 1] # transport inlet
    # ax4.set_title('transport inlet')
    # ax5 = axs[2, 0] # total shroud
    # ax5.set_title('total shroud')
    # ax6 = axs[2, 1] # total inlet
    # ax6.set_title('total inlet')
    # # SHROUD
    # plot_eff(ax1,diametros_micro,shroud_eff_asp,'b','asp shroud')
    # plot_eff(ax1,diametros_micro,shroud_eff_transm,'r','transm shroud')
    # plot_eff(ax3,diametros_micro,shroud_eff_diff,'r','difusion')
    # plot_eff(ax3,diametros_micro,shroud_eff_turb,'g','turbulento',turb=True)
    # plot_eff(ax3,diametros_micro,shroud_eff_sed,'b','sedimentacion')
    # plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD,'k','TOTAL SHROUD')
    # plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD_ASP,'cyan','SHROUD ASPIRACION')
    # plot_eff(ax5,diametros_micro,EFICIENCIAS_SHROUD_TRANSP,'m','SHROUD TRANSPORTE')
    # # INLET
    # plot_eff(ax2,diametros_micro,inlet_eff_asp,'b','asp inlet')
    # plot_eff(ax2,diametros_micro,inlet_eff_transm,'r','transm inlet')
    # plot_eff(ax4,diametros_micro,inlet_eff_diff,'r','difusion')
    # plot_eff(ax4,diametros_micro,inlet_eff_turb,'g','turbulento',turb=True)
    # plot_eff(ax4,diametros_micro,inlet_eff_sed,'b','sedimentacion')
    # plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET,'k','TOTAL INLET')
    # plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET_ASP,'cyan','INLET ASPIRACION')
    # plot_eff(ax6,diametros_micro,EFICIENCIAS_INLET_TRANSP,'m','INLET TRANSPORTE')


    fig2, ax = plt.subplots(1,1)

    ax.plot([0,50],[1,1],linestyle='dashed',color='gray')
    eff_total = ax.plot(diametros_micro, EFICIENCIAS_TOTAL,label='EFICIENCIA TOTAL ratio= {}'.format(ratio),color='k',marker='o',linestyle='',markersize=2)
    fit_exp_curve(ax,diametros_micro,EFICIENCIAS_TOTAL,color='k')
    eff_inlet = ax.plot(diametros_micro, EFICIENCIAS_INLET,label='EFICIENCIAS_INLET',color='cyan',marker='o',linestyle='',markersize=2)
    fit_exp_curve(ax,diametros_micro,EFICIENCIAS_INLET,color='cyan')
    eff_shroud = ax.plot(diametros_micro, EFICIENCIAS_SHROUD,label='EFICIENCIAS_SHROUD',color='m',marker='o',linestyle='',markersize=2)
    fit_exp_curve(ax,diametros_micro,EFICIENCIAS_SHROUD,color='m')
    eff_factor = ax.plot(diametros_micro, FACTOR,label='FACTOR',color='dimgray',marker='o',linestyle='',markersize=2)
    fit_exp_curve(ax,diametros_micro,FACTOR,color='dimgray')
    ax.plot(diametros_micro, ASPIRACION, label='ASPIRACION', color='red', marker='o', linestyle='', markersize=2)
    # --------------------------------------------------------------------------------------
    # EFICIENCIA SIPPOLA ET AL.
    # ax.plot(x_asp_5ms_sipp,asp_5ms_sipp,label='Aspiracion Sippola 8.8 m/s',linestyle='--')
    # --------------------------------------------------------------------------------------
    # EFICIENCIA McFARLAND ET AL.
    ax.plot(x_asp_farland,asp_farland,label='ASP FARLAND',linestyle='--')
    ax.plot(x_transm_farland,transm_farland,label='TRANS FARLAND',linestyle='--')
    # --------------------------------------------------------------------------------------
    ax.set_xlim(0,np.max(diametros_micro))
    # max_sipp = np.max(asp_5ms)
    # ax.set_ylim(0,np.max([maximo,max_sipp]))
    ax.legend(fontsize=15,markerscale=2,framealpha=1,loc=2)
    ax.grid()
    ax.set_xlabel(r'Particle Diameter, dp [$\mu$m]')
    ax.set_ylabel(r'Efficiency $\eta$')

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

plt.show()
print('done')


# plt.show()


