from fluids import core, ATMOSPHERE_1976
from math import exp, pi
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np

import scienceplots
sns.light_palette("seagreen", as_cmap=True)

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 600,
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.4,
    'axes.linewidth': 1.2,
    'font.family': 'serif',
})
# ----------------------------------------------------------------------------------------------------------------------
def replace_comma_with_dot(x):
    try:
        return float(x.replace(",", "."))
    except Exception:
        return x


base_dir = Path(__file__).resolve().parent
wd = base_dir / 'PLC'
file = 'asp.csv'
df = pd.read_csv(wd / file)
df_asp = df.applymap(replace_comma_with_dot)
file = 'diff.csv'
df = pd.read_csv(wd / file)
df_diff = df.applymap(replace_comma_with_dot)
file = 'sampling.csv'
df = pd.read_csv(wd / file)
df_sampling = df.applymap(replace_comma_with_dot)
file = 'sedim.csv'
df = pd.read_csv(wd / file)
df_sedim = df.applymap(replace_comma_with_dot)
file = 'codo.csv'
df = pd.read_csv(wd / file)
df_codo = df.applymap(replace_comma_with_dot)

# ----------------------------------------------------------------------------------------------------------------------
# EFICIENCIA MUESTREO
# aspiracion
def n_asp(u0_u,stokes):
    n_aspiracion =  1 + (u0_u-1)*(1-(1+stokes*(2+0.617*(1/u0_u)))**-1) #valido para 0.005 < stokes < 2.03
    # n_aspiracion =  1 + (u0_u-1)*(1-(1+3.77*stokes**(0.883))**-1) #valido para 0.005 < stokes < 10
    return n_aspiracion


# transmision
def n_trans(u0_u,stokes):
    n_transport_inert = (1 + (u0_u - 1) / (1 + 2.66 / stokes ** (2 / 3))) / (1 + (u0_u - 1) / (1 + 0.418 / stokes))
    return n_transport_inert

# factor correccion Gong et al.
def factor_corr(u0_u,stokes):
    # stokes = core.Stokes_number(V, dp, diam, rho_p, atmos.mu) #stokes del shroud
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

    lambda_r = 0.0664  # esta linea
    lambda_r = lambda_r *10**-6
    kn = 2*lambda_r/dp
    return kn

def Cc(dp, P, T):
    P = P * 10 ** -3
    dp = dp/ 10 ** -6
    cc = 1 + 1*(15.60+7*np.exp(-0.059*P*dp))/(P*dp)

    # kn_2 =  2* 0.6*10**-7/dp#free path (0.6 – 0.7) 10 -7
    cc_2 = 1+ Kn(dp,P,T)*(1.142+0.558*exp(-0.999/Kn(dp,P,T)))
    return cc_2


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
    Vts = tau_rel(rho_p,dp,mu,P,T)*9.81*Cc(dp,atmos.P,atmos.T)
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
    # print(dp, stokes)
    tau_mas = 0.0395*stokes*re**(3/4)
    if tau_mas > 12.9:
        V_mas = 0.1
    else:
        V_mas = 6*10**-4*tau_mas**2+2*10**-8*re
    Vt = V_mas*V/(5.03*re**(1/8))
    return np.exp(-(pi*d*L*Vt)/(Q))


def n_iner_codo(V, dp, d, rho_p,ang):
    """

    Args:
        Stks: numero de stokes
        ang: angulo de curvatura en grados

    Returns: rendimiento inercial debido a un codo

    """
    # Stks = core.Stokes_number(V, dp, d, rho_p, atmos.mu)
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
    Stks = core.Stokes_number(V, dp, d, rho_p, atmos.mu)*Cc(dp,atmos.P,atmos.T)
    eff = exp(-2.823*Stks*ang*pi/180)

    # eff =  1 - Stks*ang*pi/180
    # if eff <0:
    #     eff = 0

    # val_1 = (Stks/0.171)
    # val_2 = (0.452*(Stks/0.171)+2.242)
    # val_3 = val_1**val_2
    # val_4 = -(2/pi*ang)
    # eff = (1+val_3)**val_4
    return eff


# ----------------------------------------------------
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
codo = []
particulas = np.linspace(0.1,50,1000)*10**-6
U0_U = U0/U_inlet
for dp in particulas:
    stk = core.Stokes_number(U_inlet,dp,d_inlet,rho_p,atmos.mu)
    stk = core.Stokes_number(U0,dp,d_inlet,rho_p,atmos.mu)*Cc(dp,atmos.P,atmos.T)
    asp_val = n_asp(U0_U,stk)
    trans_val = n_trans(U0_U,stk)
    samp_val = asp_val * trans_val

    turb_val_1 = n_turb_inert(d_inlet_1,L_inlet_1,Q_inlet,Re_inlet,U_inlet,dp)
    sed_val_1 = n_sedim(d_inlet_1,L_inlet_1,rho_p,dp,atmos.mu,atmos.P,atmos.T,Q_inlet,U_inlet,Re_inlet)
    diff_val_1 = n_dif(dp,atmos.P,atmos.T,L_inlet_1,Q_inlet,atmos.mu,Re_inlet,atmos.rho)

    codo.append(n_iner_codo(U_inlet,dp,d_inlet_1,rho_p,60))

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

particulas = np.linspace(0.1,50,1000)

# plt.plot(particulas,asp,label='asp',color='b')
# plt.plot(particulas,trans,label='trans')
fig2, ax = plt.subplots(1, 1, figsize=(15, 9), constrained_layout=True)

color_modelo = {
    'muestreo': '#C00000',
    'sedimentacion': '#D4A017',
    'turbulenta': '#1B9E77',
    'difusion': '#1F77B4',
    'codo': '#E67E22',
    'total': '#1A1A1A',
}

ax.plot(particulas, samp, label='Muestreo - Modelo', color=color_modelo['muestreo'])
ax.plot(particulas, sed, label='Sedimentacion - Modelo', color=color_modelo['sedimentacion'])
ax.plot(particulas, turb, label='Deposicion turbulenta - Modelo', color=color_modelo['turbulenta'])
ax.plot(particulas, inlet, label='Eficiencia total del inlet - Modelo', color=color_modelo['total'], linewidth=2.8)
ax.plot(particulas, diff, label='Difusion - Modelo', color=color_modelo['difusion'])
ax.plot(particulas, codo, label='Perdidas en codo - Modelo', color=color_modelo['codo'])
# plt.xscale('log')


ax.plot(df_asp['x'], df_asp['Curve1'], color=color_modelo['muestreo'], linestyle='--', linewidth=2.0,
        label='Aspiracion - PLC')
ax.plot(df_diff['x'], 1-df_diff['Curve1'], color=color_modelo['difusion'], linestyle='--', linewidth=2.0,
        label='Difusion - PLC')
ax.plot(df_sedim['x'], 1-df_sedim['Curve1'], color=color_modelo['sedimentacion'], linestyle='--', linewidth=2.0,
        label='Sedimentacion - PLC')
ax.plot(df_sampling['x'], df_sampling['Curve1'], color=color_modelo['total'], linestyle='--', linewidth=2.0,
        label='Muestreo total - PLC')
ax.plot(df_codo['x'], 1-df_codo['Curve1'], color=color_modelo['codo'], linestyle='--', linewidth=2.0,
        label='Codo - PLC')




cmap = plt.get_cmap('tab20')
num_colors = 19
colors_fill = [cmap(i) for i in np.linspace(0, 1, num_colors)]

plot_bands = True
band_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.1,3,5,7.5,10,12.5,15,17.5,20,22,30,40,50]
banda_fill = []
maximo = np.max(asp)
if plot_bands:
    from matplotlib.legend import Legend
    for i in range(len(band_list)-1):
        rango=[band_list[i],band_list[i+1]]
        fill = ax.fill_between(rango, 0, maximo, color=colors_fill[i], alpha=0.10, zorder=0)
        banda_fill.append(fill)
    legend2=Legend(ax, banda_fill, ['{} - {}'.format(band_list[i], band_list[i+1]) for i in range(len(band_list)-1)],
                   title='Rangos LOAC [um]', fontsize=11, title_fontsize=12, framealpha=0.96,
                   frameon=True, loc='lower right', bbox_to_anchor=(0.98, 0.02),
                   borderaxespad=0.2, ncol=2)
    ax.add_artist(legend2)

style_handles = [
    Line2D([0], [0], color='k', lw=2.5, linestyle='-', label='Modelo teorico'),
    Line2D([0], [0], color='k', lw=2.5, linestyle='--', label='Datos digitalizados PLC'),
]
legend_estilo = ax.legend(handles=style_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                          frameon=True, title='Tipo de curva', fontsize=14, title_fontsize=14,
                          borderaxespad=0.3)
ax.add_artist(legend_estilo)

ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.56), frameon=True, title='Mecanismo',
          fontsize=14, title_fontsize=15, borderaxespad=0.35)
ax.set_xlim(0,50)
ax.set_ylim(0,1.75)
ax.set_xlabel('Diametro de particula, $d_p$ [um]')
ax.set_ylabel('Eficiencia [-]')
ax.set_title('Validacion del metodo de eficiencia del inlet frente a resultados PLC')
ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.30)

# ------------------------------------------------------------------------------------------------------
# ZOOM



def axis_data_transform(parentfig, srcAx, xin, yin, inverse=False):
    xy = parentfig.transFigure.inverted().transform(
        srcAx.transData.transform((xin, yin)))
    return xy[0], xy[1]


def addzoomplot(fig, srcAx, desLoc=[], srcLoc=[], conVec=[], color='r', linewidth=1,
                showconnectors=True, showhighlightbox=True, showticks=True):
    """
    Adds a zoom axes inside the given figure and axes.

    Call signatures::


        addzoomplot(fig, srcAx)        # adds a zoom axes inside fig and srcAx interactively
        ax = addzoomplot(fig, srcAx)   # adds a zoom axes inside fig and srcAx interactively and returns the zoom axis


    Parameters:

    - fig (figure):The figure the zoom axes will be inserted in.

    - srcAx (axis): The axis inside fig that the zoom axes will be inserted in

    - srcLoc = [xmin, ymin, xmax, ymax]

    - desLoc = [bl_x, bl_y, tr_x, tr_y]

    - conVec = [parentloc1=3, parentloc2=4, zoomloc1=1, zoomloc2=2]

    -- xmin, xmax (values): The minimum and maximum values for the xlimit on the zoom axis

    -- ymin, ymax (values): The minimum and maximum values for the ylimit on the zoom axis

    -- bl_x, bl_y (values): The bottom left corner of the zoom axis in the data coordinate of the srcAx
                             e.g., if you want the zoom axis to appear with the bottom left corner at x=2.5 and
                                   y=3 on your parent axis you just set the bl_x = 2.5 and bl_y = 3 and the
                                   function will automatically convert the coordiantes to srcAx normalized
                                   values between 0 and 1

                                   It is important to call tight_layout() before you add any zoomplots using this
                                   function for this coordinate conversion to work properly, otherwise it will not
                                   work as expected.

    -- tr_x, tr_y (values): The top right corner of the zoom axis in the data coordinate of the srcAx
                             e.g., if you want the zoom axis to appear with top right corner at x=2.5 and y=3
                                   on your parent axis you just set the bl_x = 2.5 and bl_y = 3 and the
                                   function will automatically convert the coordiantes to srcAx normalized
                                   values between 0 and 1

                                   It is important to call tight_layout() before you add any zoomplots using this
                                   function for this coordinate conversion to work properly, otherwise it will not
                                   work as expected.

    -- parentloc1, parentloc2 ({1,2,3,4}): The start location of the connections from the box that highlights the
                                            limits of the zoom on the srcAx with xmin, xmax, ymin and ymax values
                                            to the zoom axes added to the plot with the function
                                            1: top left corner
                                            2: top right corner
                                            3: bottom right corner
                                            4: bottom left corner

    -- zoomloc1, zoomloc2 ({1,2,3,4}): The end location of the connections from the box that highlights the
                                            limits of the zoom on the srcAx with xmin, xmax, ymin and ymax values
                                            to the zoom axes added to the plot with the function
                                            1: top left corner
                                            2: top right corner
                                            3: bottom right corner
                                            4: bottom left corner

    - color (color value or string): The color for the highliught box and the connection lines

    - linewidth (decimal): The linewidth (thickness) for the highlight box and connectors

    - showconnectors (bool): The boolean to set whether to show the connector lines or not

    - showhighlightbox (bool): The boolean to set wether to show the highlight box or not

    - showticks (bool): The boolean to set whether to show the ticks on x and y axis of the added zoom axis
    Returns:

        Added zoom plot axis (Axis)

    Notes:

    If you call plt.tight_layout() on your plot or figure for the figure
    you are working with, it is important to call it before you call the
    addzoomplot function as the dimensions will not be proper otherwise
    """
    # This functions helps add a zoom sub axis to your existing plot
    # It is very useful when you need to add magnieifed portions to your plot
    # to give your viewer a better understanding of what is going on in a
    # region of the plot that is very crowded

    if (srcLoc == []):
        prompt = plt.text(0, 1, 'Step 1: Select SOURCE rectangle, 1st bottom-left then top-right corner',
                          horizontalalignment='left',
                          verticalalignment='top',
                          color='red',
                          transform=srcAx.transAxes)
        pts = plt.ginput(2)
        srcLoc = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
        prompt.remove()

    xmin = srcLoc[0]
    ymin = srcLoc[1]
    xmax = srcLoc[2]
    ymax = srcLoc[3]

    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=linewidth, edgecolor=color,
                             facecolor='none')
    if (showhighlightbox):
        srcAx.add_patch(rect)

    if (desLoc == []):
        prompt = plt.text(0, 1, 'Step 2: Select DESTINATION rectangle, 1st bottom-left then top-right corner',
                          horizontalalignment='left',
                          verticalalignment='top',
                          color='blue',
                          transform=srcAx.transAxes)
        pts = plt.ginput(2)
        desLoc = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
        prompt.remove()
    bl_x = desLoc[0]
    bl_y = desLoc[1]
    tr_x = desLoc[2]
    tr_y = desLoc[3]

    if (conVec == []):
        srcCenter = [(srcLoc[0] + srcLoc[2]) / 2, (srcLoc[1] + srcLoc[3]) / 2]
        desCenter = [(desLoc[0] + desLoc[2]) / 2, (desLoc[1] + desLoc[3]) / 2]
        if (srcCenter[0] < desCenter[0]):
            if (desCenter[1] > srcLoc[1] and desCenter[1] < srcLoc[3]):
                conVec = [2, 3, 1, 4]
            elif (desCenter[1] < srcLoc[1]):
                conVec = [3, 4, 2, 1]
            elif (desCenter[1] > srcLoc[1]):
                conVec = [1, 2, 4, 3]
        else:
            if (desCenter[1] > srcLoc[1] and desCenter[1] < srcLoc[3]):
                conVec = [1, 4, 2, 3]
            elif (desCenter[1] < srcLoc[1]):
                conVec = [3, 4, 2, 1]
            elif (desCenter[1] > srcLoc[1]):
                conVec = [1, 2, 4, 3]

    parentloc1 = conVec[0]
    parentloc2 = conVec[1]
    zoomloc1 = conVec[2]
    zoomloc2 = conVec[3]

    bb_data = Bbox.from_bounds(bl_x, bl_y, tr_x - bl_x, tr_y - bl_y)
    disp_coords = srcAx.transData.transform(bb_data)
    fig_coords = fig.transFigure.inverted().transform(disp_coords)

    zoomax = fig.add_axes(Bbox(fig_coords))

    axblx, axbly = axis_data_transform(fig, srcAx, bl_x, bl_y)
    axtrx, axtry = axis_data_transform(fig, srcAx, tr_x, tr_y)
    for l in srcAx.get_lines():
        zoomax.plot(l.get_data()[0], l.get_data()[1], linestyle=l.get_linestyle(),
                    markevery=l.get_markevery(), color=l.get_color(),
                    markersize=l.get_markersize(), label=l.get_label(),
                    antialiased=l.get_antialiased(),zorder=10)

    bblx, bbly = axis_data_transform(fig, srcAx, xmin, ymin)
    btrx, btry = axis_data_transform(fig, srcAx, xmax, ymax)

    if parentloc1 == 1:
        x1 = bblx
        y1 = btry
    elif parentloc1 == 2:
        x1 = btrx
        y1 = btry
    elif parentloc1 == 3:
        x1 = btrx
        y1 = bbly
    else:
        x1 = bblx
        y1 = bbly

    if zoomloc1 == 1:
        x2 = axblx
        y2 = axtry
    elif zoomloc1 == 2:
        x2 = axtrx
        y2 = axtry
    elif zoomloc1 == 3:
        x2 = axtrx
        y2 = axbly
    else:
        x2 = axblx
        y2 = axbly

    if (showconnectors):
        srcAx.annotate("",
                       xy=(x1, y1), xycoords='figure fraction',
                       xytext=(x2, y2), textcoords='figure fraction',
                       arrowprops=dict(arrowstyle="-",
                                       connectionstyle="arc3,rad=0",
                                       color=color,
                                       linewidth=linewidth,zorder=10))

    if parentloc2 == 1:
        x1 = bblx
        y1 = btry
    elif parentloc2 == 2:
        x1 = btrx
        y1 = btry
    elif parentloc2 == 3:
        x1 = btrx
        y1 = bbly
    else:
        x1 = bblx
        y1 = bbly

    if zoomloc2 == 1:
        x2 = axblx
        y2 = axtry
    elif zoomloc2 == 2:
        x2 = axtrx
        y2 = axtry
    elif zoomloc2 == 3:
        x2 = axtrx
        y2 = axbly
    else:
        x2 = axblx
        y2 = axbly

    if (showconnectors):
        srcAx.annotate("",
                       xy=(x1, y1), xycoords='figure fraction',
                       xytext=(x2, y2), textcoords='figure fraction',
                       arrowprops=dict(arrowstyle="-",
                                       connectionstyle="arc3,rad=0",
                                       color=color,
                                       linewidth=linewidth))

    plt.setp(zoomax, xlim=(xmin, xmax), ylim=(ymin, ymax))

    if (showticks == False):
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labeltop=False)  # labels along the top edge are off
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the left edge are off
            right=False,  # ticks along the right edge are off
            labelleft=False,  # labels along the left edge are off
            labelright=False)  # labels along the right edge are off

    return zoomax


srcLoc = [0.2,0.996 ,1.5,1.01]
desLoc = [0.35, 0.04, 9.5, 0.40]
zoom = addzoomplot(fig2,ax,desLoc,srcLoc,color='k',showhighlightbox=True)

plot_bands = True
band_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.1,3,5,7.5,10,12.5,15,17.5,20,22,30,40,50]
banda_fill = []
maximo = np.max(asp)
if plot_bands:
    from matplotlib.legend import Legend
    for i in range(len(band_list)-1):
        rango=[band_list[i],band_list[i+1]]
        fill = zoom.fill_between(rango, 0, maximo, color=colors_fill[i], alpha=0.12)
        banda_fill.append(fill)

zoom.set_yticklabels([])
zoom.set_title('Detalle en particulas submicronicas', fontsize=12)
zoom.set_xlabel('$d_p$ [um]', fontsize=11)
zoom.tick_params(axis='both', labelsize=10)
zoom.grid(True, linestyle='--', linewidth=0.5, alpha=0.25)

nombre_base = 'validacion_metodo_plc_publicacion'
fig2.savefig(base_dir / f'{nombre_base}.png', bbox_inches='tight')
fig2.savefig(base_dir / f'{nombre_base}.pdf', bbox_inches='tight')
plt.show()
