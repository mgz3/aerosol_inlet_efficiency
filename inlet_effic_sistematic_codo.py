from math import exp, sqrt, pi,cos,tan
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['science','ieee'])
fig, ax = plt.subplots()

To = 288.15  # K
alpha_T = 6.5 * 10 ** -3  # K/m
Po = 1.01325 * 10 ** 5  # N/m**2
Rho_o = 1.225  # kg/m**3
g = 9.80665
R_a = 287.05  # constante del aire  J/(KgK)
Mu_o = 1.716 * 10 ** -5  # N.s/m**2
Su = 111.  # constante de Sutherland K
k_boltzman = 1.380649*10**-23 #N.m/K

def T(h):
    return To - alpha_T * h

def P(h):
    return Po * (1 - alpha_T * h / To) ** (g / (R_a * alpha_T))


def rho_h(h):
    return Rho_o * (1 - alpha_T * h / To) ** ((g / (R_a * alpha_T)) - 1)


def Mu(T):
    return Mu_o * ((T / To) ** (3 / 2) * ((To + Su) / (T + Su)))


def Re(rho, u, L, mu):
    return rho * L * u / mu


def LAMBDA(P, T):
    """

    Args:
        P: presion atmosferica en Pa
        T: temperatura en K

    Returns: valor de camino medio libre para esas variables termodinamicas

    """
    P = P * 10 ** -3
    lambda_r = 0.0664 * 10**-6
    return lambda_r * (101 / P) * (T / 293) * ((1 + 110 / 293) / (1 + 110 / T))


def Kn(dp, P, T):
    return 2*0.0664 * 10**-6/dp
    # return 2 * LAMBDA(P, T) / dp


def Cc(dp, P, T):
    alpha = 1.142
    beta = 0.558
    gamma = 0.999
    return 1 + Kn(dp, P, T) * (alpha + beta * exp(-gamma / Kn(dp, P, T)))

def Cc_2(dp, P, T):
      dp_micro = dp * 10**6
      p_kpa = P *10**-3
      alpha = 15.6
      beta = 7
      gamma = 0.059
      return 1 + 1/(p_kpa*dp_micro) * (alpha + beta * exp(-gamma / (p_kpa*dp_micro)))

def tau_rel(rho_p, dp, mu, P, T):
    # print('rho_p: ',rho_p)
    # print('dp: ',dp)
    # print('mu: ',mu)
    # print('cc: ',Cc(dp, P, T))

    return rho_p * dp**2 * Cc(dp, P, T) / (18 * mu)


def stks(u, d, rho_p, dp, mu, P, T):
    """

    Args:
        u: velocidad --> m/s
        d: longitud de referencia --> m
        rho_p: densidad de la particula --> kg/m**3
        dp: diametro de la particula --> m
        mu: viscosidad dinamica --> Pa.s
        P: presion atmosferica --> Pa
        T: temperatura ambiente --> K

    Returns: numero de stokes

    """
    return tau_rel(rho_p, dp, mu, P, T) * u / d


def stokes_particle(dp,P,T,rho_p,u0,mu_h,d):
    """

    Args:
        dp: diametro de la particula
        P: presion
        T: temperatura
        rho_p: densidad de la particula
        u0: velocidad de la corriente libre
        mu_h: viscosidad dinamica
        d: diametro interno

    Returns: numero de stokes de la particula

    """
    kn = Kn(dp,P,T)
    cc = 1+ kn*(1.142+0.558*exp(-0.999/kn))
    stokes_part = (dp**2)*rho_p*Cc_2(dp,P,T)*u0/(18*mu_h*d)
    return stokes_part
# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA DE MUESTREO
# aspiracion

def k(u_uo):
    return 2 + 0.617 * (1 / u_uo)  # <-------------------------------------------REVISAR U/U0


def n_asp(Stks,u0_u):
    n_aspiracion = 1
    # n_aspiracion = 1 + (u0_u - 1) * (1 - 1 / (1 + k(u0_u) * Stks)) # rango mas chico de aplicabilidad para #stokes
    if 0.005 < Stks < 10:
        n_aspiracion = 1 + (u0_u - 1) * (1 - 1 / (1 + 3.77* Stks**0.883))  #valido para 0.005 < stokes < 10
    elif 0.01 < Stks < 100:
        n_aspiracion = 1 + (u0_u - 1)/(1+0.418/Stks)
    return n_aspiracion


# transmision

def n_trans(Stks,u0_u):
    n_transport_inert = (1 + (u0_u - 1) / (1 + 2.66 * Stks ** (-2 / 3))) / (1 + (u0_u - 1) / (1 + 0.418 / Stks))
    return n_transport_inert


# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA DE TRANSPORTE

# difusion

def n_dif(dp,P,T,L,Q,mu,rho_d,re,rho_f):
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
    n = exp(-zita * Sh)
    return n

# sedimentacion

def n_sedim(d,L,rho_p,dp,mu,P,T,Q,ang=0):
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

    Returns:

    """
    Vts = tau_rel(rho_p,dp,mu,P,T)*g

    return exp(-1*(d*L*Vts*cos(ang*pi/180))/Q)

# deposicion inercial turbulenta

def n_turb_inert(d,L,Q,Stks,re,u):
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
    tau_mas = 0.0395*Stks*re**(3/4)
    if tau_mas > 12.9:
        V_mas = 0.1
    else:
        V_mas = 6*10**-4*tau_mas**2+2*10**-8*re
    Vt = V_mas*u/(5.03*re**(1/8))
    # Vt = (6*10**-4*(0.0395*Stks*re**(3/4))**2+2*10**(-8)*re)/(5.03*re**(1/8))*u
    return exp(-(pi*d*L*Vt)/(Q))

# deposicion inercial codo

def n_iner_codo(Stks,ang):
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

    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    print('valores del codito')
    if Stks > 1:
        print('ACAAAAAAA ESTUPIDOOOOO   ---------------------------------------------------1231234-2341234-13-423-4-23')
    print('stokes : ',Stks)
    print('ang en grados : {}    ang en radianes: {}'.format(ang,ang*pi/180))
    print('-.-.-.--.-.-.-.-.-.-.-.-.-.-.')
    return exp(-2.823*Stks*ang*pi/180)
    # return efficiency_bend

def Lenght(d):
    """
    Args:
        d: diametro inicial en mm

    Returns: largo del primer tramo de la seccion para un
                angulo de difusor de 7º en mm
    """
    ang = 7 * pi / 180
    L = (24-d)/(tan(ang))
    return L

def V_2(v1,d):
    """
    Args:
        v1: v1 en m/s
        d: diametro inicial en mm
    Returns:

    """
    v_2 = v1*(d/24)**2
    return v_2

v1 = 25  # m/s
diam_2 = 24  # diametro interno de la camara en mm
X = 4.4  # distancia a la camara (cm) para la formula de espesor de desplazamiento

A = pi * 12 ** 2  # area camara ensayo mm*2

diametros_iniciales = [10,8,6,4] #diametros iniciales en mm

# CALCULO DE VARIABLES AERODINAMICAS
h = 1000  # m
p = P(h)
t = T(h)
rho = rho_h(h)
mu = Mu(t)

rho_p = 1000  # kg/m**3    SUPOSICION
# datos para el calculo de la sonda completa
diametro_prueba_part = 10 ** -6  # m
diam_1 = 8 # eleccion de diam_1 en mm
ANGS = [0,15,30,45,60,75,90]

dash = ['solid','dotted','dashed','dashdot', (0, (5, 10)),(0, (1, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
i_dash = 0

STOKES_EXT = []
STOKES_INT = []

ang_rot = 0
diam_1 = 8 # eleccion de diam_1 en mm
v1 = 25
HEIGHTS = [0,100,500,1000,1500,2000,2500,3000]

VELOCIDADES = [5,10,15,20,25]

# for v1 in VELOCIDADES:
# for h in HEIGHTS:
for ang_rot in ANGS:
    p = P(h)
    t = T(h)
    rho = rho_h(h)
    mu = Mu(t)

    re = Re(rho, v1, diam_1 * 10 ** -3, mu)

    v2 = V_2(v1,diam_1)
    esp_desp = 1.74 * sqrt(0.17 * X / (v2 * 100))  # cm
    Aeff = pi * (diam_2 / 2 - esp_desp * 10) ** 2  # area efectiva camara de ensayo  mm**2
    Um = v2  # velocidad en la camara de ensayo, se supone igual a la calculada teoricamente
    R = 5  # relacion de areas y/o velocidades, de la ecuacion de continuidad
    U = Um * R * (Aeff / A)  # velocidad media a la entrada de la sonda
    U0 = v1
    U0_U = U0 / U
    U0_U = 1


    # calculo sonda externa
    # datos
    # tramo I
    diam_I_ext = diam_1 * 10 ** -3  # mm diametro interno de entrada de la sonda
    # LI_ext = 108 * 10 ** -3  # mm long tramo I
    LI_ext = Lenght(diam_1) * 10 ** -3  # mm long tramo I
    VI_ext = v1  # m/s velocidad en diam_1
    QI_ext = VI_ext * (pi * (diam_I_ext) ** 2 / 4)  # m**3/s  caudal volumetrico en I
    # tramo II
    diam_II_ext = diam_2 * 10 ** -3  # mm diametro interno de la camara
    LII_ext = 44 * 10 ** -3  # mm long tramo II
    VII_ext = v2  # m/s velocidad en diam_2
    QII_ext = VII_ext * (pi * (diam_II_ext) ** 2 / 4)  # m**3/s  caudal volumetrico en I

    # calculo de la sonda interna
    # tramo I
    diam_I_int = 8*10**-3                      # mm diametro interno de entrada de la sonda
    LI_int= 30*10**-3                              # mm long tramo I
    VI_int = v2                              # m/s velocidad en diam_1
    QI_int =VI_int*(pi*(diam_I_int)**2/4)     #m**3/s  caudal volumetrico en I
    # ang_rot = 0                            #ang de rotacion en grados
    # tramo II
    diam_II_int =  8*10**-3                  # mm diametro interno de la camara
    LII_int= 40*10**-3                              # mm long tramo II
    VII_int = v2                            # m/s velocidad en diam_2
    QII_int =VI_int*(pi*(diam_II_int)**2/4)     #m**3/s  caudal volumetrico en I

    # diametros_micro = [0.25,0.35,0.45,0.55,0.65,0.8,1,(1.1+3)/2,(3+5)/2,(5+7.5)/2,(7.5+10)/2,(10+12.5)/2,(12.5+15)/2,(15+17.5)/2,(17.5+20)/2,(20+22)/2,(22+30)/2,(30+40)/2,(40+50)/2]
    diametros_micro = np.linspace(0,50,1000)
    DIAMETROS = []
    EFICIENCIAS = []

    for i in range(len(diametros_micro)):
        DIAMETROS.append(diametros_micro[i] * 10 ** -6)  # cambio de unidad, estaban en micrometros

    for diametro_prueba_part in DIAMETROS:
        stokes = stokes_particle(diametro_prueba_part, p, t, rho_p, v1, mu, diam_1 * 10 ** -3)
        # calculo de la sonda externa
        # eficiencia aspiracion

        n_aspiracion_ext = n_asp(stokes,U0_U)
        n_trans_ext=n_trans(stokes,U0_U)
        #eficiencia de transporte
        #TRAMO I
        N_DIF_I_ext=n_dif(diametro_prueba_part,p,t,LI_ext,QI_ext,mu,rho_p,re,rho)
        N_SED_I_ext=n_sedim(diam_I_ext,LI_ext,rho_p,diametro_prueba_part,mu,p,t,QI_ext)
        N_TURB_INER_I_ext=n_turb_inert(diam_I_ext,LI_ext,QI_ext,stokes,re,VI_ext)
        #TRAMO II
        N_DIF_II_ext=n_dif(diametro_prueba_part,p,t,LII_ext,QII_ext,mu,rho_p,re,rho)
        N_SED_II_ext=n_sedim(diam_II_ext,LII_ext,rho_p,diametro_prueba_part,mu,p,t,QII_ext)
        N_TURB_INER_II_ext=n_turb_inert(diam_II_ext,LII_ext,QII_ext,stokes,re,VII_ext)

        N_SAMPLING_EXT = n_aspiracion_ext*n_trans_ext
        N_TRANSPORT_EXT = N_DIF_I_ext*N_SED_I_ext*N_TURB_INER_I_ext*N_DIF_II_ext*N_SED_II_ext*N_TURB_INER_II_ext
        N_PROBE_EXT = N_SAMPLING_EXT*N_TRANSPORT_EXT
        # ----------------------------------------------------------------------------------------------------------------------

        # calculo de la sonda interna
        # eficiencia aspiracion
        stokes_int = stokes_particle(diametro_prueba_part, p, t, rho_p, v2, mu, diam_I_int)
        # stokes_int = stokes
        U0_U_int=1      #relacion uo/u, se supone 1 porque no tenemos manera rapida de aproximarla ni medirla
        n_aspiracion_int = n_asp(stokes_int,U0_U_int)
        n_trans_int=n_trans(stokes_int,U0_U_int)
        #eficiencia de transporte
        #TRAMO I
        N_DIF_I_int=n_dif(diametro_prueba_part,p,t,LI_int,QI_int,mu,rho_p,re,rho)
        N_SED_I_int=n_sedim(diam_I_int,LI_int,rho_p,diametro_prueba_part,mu,p,t,QI_int)
        N_TURB_INER_I_int=n_turb_inert(diam_I_int,LI_int,QI_int,stokes_int,re,VI_int)
        #TRAMO II
        N_DIF_II_int=n_dif(diametro_prueba_part,p,t,LII_int,QII_int,mu,rho_p,re,rho)
        N_SED_II_int=n_sedim(diam_II_int,LII_int,rho_p,diametro_prueba_part,mu,p,t,QII_int)
        N_TURB_INER_II_int=n_turb_inert(diam_II_int,LII_int,QII_int,stokes_int,re,VII_int)
        N_CODO_int= n_iner_codo(stokes_int,ang_rot)

        N_SAMPLING_int = n_aspiracion_int*n_trans_int
        N_TRANSPORT_int = N_DIF_I_int*N_SED_I_int*N_TURB_INER_I_int*N_DIF_II_int*N_SED_II_int*N_TURB_INER_II_int*N_CODO_int
        N_PROBE_int = N_SAMPLING_int*N_TRANSPORT_int

          # ----------------------------------------------------------------------------------------------------------------------
          # calculo de eficiencia total
        N_PROBE_TOTAL = N_PROBE_EXT*N_PROBE_int
        # N_PROBE_TOTAL = N_PROBE_int    #--------------------------este valor es solo para probar
        EFICIENCIAS.append(N_PROBE_TOTAL)
        print('--------------------------------------------------------')
        print('Diametro: {} microm\n'.format(diametro_prueba_part / (10 ** -6)))
        print("SONDA EXTERNA \n"
              "Eficiencia de muestreo\n"
              "n_asp = {}\n"
              "n_trans = {}\n"
              "Eficiencia de transporte\n"
              "TRAMO I\n"
              "n_dif = {}\n"
              "n_sed = {}\n"
              "n_turb_inert = {}\n"
              "TRAMO II\n"
              "n_dif = {}\n"
              "n_sed = {}\n"
              "n_turb_inert = {}\n".format(n_aspiracion_ext,n_trans_ext,N_DIF_I_ext,N_SED_I_ext
                                           ,N_TURB_INER_I_ext,N_DIF_II_ext,N_SED_II_ext,N_TURB_INER_II_ext))
        N_SAMPLING_EXT = n_aspiracion_ext*n_trans_ext
        N_TRANSPORT_EXT = N_DIF_I_ext*N_SED_I_ext*N_TURB_INER_I_ext*N_DIF_II_ext*N_SED_II_ext*N_TURB_INER_II_ext
        N_PROBE_EXT = N_SAMPLING_EXT*N_TRANSPORT_EXT

        print('..........................')
        print('EFICIENCIAS TOTALES\n'
              'Eficiencia Muestreo = {}\n'
              'Eficiencia transporte = {}\n'
              'Eficiencia de la sonda = {}\n'.format(N_SAMPLING_EXT,N_TRANSPORT_EXT,N_PROBE_EXT))
        print('...............................')

        print("SONDA INTERNA \n"
              "Eficiencia de muestreo\n"
              "n_asp = {}\n"
              "n_trans = {}\n"
              "Eficiencia de transporte\n"
              "TRAMO I\n"
              "n_dif = {}\n"
              "n_sed = {}\n"
              "n_turb_inert = {}\n"
              "TRAMO II\n"
              "n_dif = {}\n"
              "n_sed = {}\n"
              "n_turb_inert = {}\n"
              "n_cod = {}".format(n_aspiracion_int,n_trans_int,N_DIF_I_int,N_SED_I_int
                                          ,N_TURB_INER_I_int,N_DIF_II_int,N_SED_II_int,N_TURB_INER_II_int,N_CODO_int))
              # "n_turb_inert = {}\n".format(n_aspiracion_int,n_trans_int,N_DIF_I_int,N_SED_I_int
              #                              ,N_TURB_INER_I_int,N_DIF_II_int,N_SED_II_int,N_TURB_INER_II_int))
              # "n_turb_inert = {}\n"
              # "n_cod = {}".format(n_aspiracion_int,n_trans_int,N_DIF_I_int,N_SED_I_int
              #                             ,N_TURB_INER_I_int,N_DIF_II_int,N_SED_II_int,N_TURB_INER_II_int,N_CODO_int))
        #
        N_SAMPLING_int = n_aspiracion_int*n_trans_int
        N_TRANSPORT_int = N_DIF_I_int*N_SED_I_int*N_TURB_INER_I_int*N_DIF_II_int*N_SED_II_int*N_TURB_INER_II_int*N_CODO_int
        N_PROBE_int = N_SAMPLING_int*N_TRANSPORT_int

        print('..........................')
        print('EFICIENCIAS TOTALES\n'
              'Eficiencia Muestreo = {}\n'
              'Eficiencia transporte = {}\n'
              'Eficiencia de la sonda = {}\n'.format(N_SAMPLING_int,N_TRANSPORT_int,N_PROBE_int))
        label = str(ang_rot) + ' °'

        # label = str(v1) + ' m/s'
        #
        # label = 'V= {}m/s'.format(v1)
        # label = '{} m'.format(h)

    ax.plot(diametros_micro, EFICIENCIAS,linestyle=dash[i_dash], label=label)
    i_dash += 1




plt.rcParams['text.usetex'] = True
ax.autoscale(tight=True)
ax.set_xlabel(r'Particle Diameter, dp (\textmu m)',fontsize=10)
ax.set_ylabel(r'Efficiency $\eta$',fontsize=10)
# ax.legend(loc='best',bbox_to_anchor=(0.3, 0.3, 0.5, 0.5))#,bbox_to_anchor=(-0.1, -0.1))
ax.legend(loc='upper right',fontsize=8,ncol=2)
ax.grid()
ax.set_xlim([0,50])
ax.set_ylim([0,1])
import os

path = 'C:\\Users\\mauro\\Desktop\\'
os.chdir(path)
# fig.savefig('eff_diff_vel_{}ms.pdf'.format(v1))
fig.savefig('eff_diff_ang_{}ms.jpg'.format(v1), dpi=300)
# fig.savefig('eff_diff_vel.jpg', dpi=300)
# fig.savefig('eff_altitude.pdf')
# fig.savefig('eff_diff_vel.pdf')
# fig.savefig('eff_altitude.jpg', dpi=300)
plt.show()
