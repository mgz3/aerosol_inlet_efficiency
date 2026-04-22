import csv
from math import exp, pi
from pathlib import Path

import numpy as np
from fluids import ATMOSPHERE_1976, core
from matplotlib import pyplot as plt

import scienceplots  # noqa: F401


plt.style.use(["science", "no-latex"])
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.4,
        "axes.linewidth": 1.2,
        "font.family": "serif",
    }
)

K_BOLTZMANN = 1.380649e-23
G = 9.81
DIFF_REL_PLOT_MAX_UM = 30.0
DIFF_REL_PLOT_YLIM_PERCENT = 20.0
TABLE_PARTICLE_RANGES_UM = [
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.85),
    (0.85, 1.0),
    (1.0, 1.2),
    (2.0, 3.0),
    (3.0, 5.0),
    (5.0, 7.5),
    (7.5, 10.0),
    (10.0, 12.5),
    (12.5, 15.0),
    (15.0, 17.5),
    (17.5, 20.0),
    (20.0, 22.0),
    (22.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
]


def n_asp(u0_u, stokes):
    return 1 + (u0_u - 1) * (1 - (1 + stokes * (2 + 0.617 / u0_u)) ** -1)


def n_trans(u0_u, stokes):
    return (1 + (u0_u - 1) / (1 + 2.66 / stokes ** (2 / 3))) / (
        1 + (u0_u - 1) / (1 + 0.418 / stokes)
    )


def factor_corr(u_u0, stokes, re):
    u0_u = 1 / u_u0
    factor = 1 - (u0_u - 1) * 0.861 * stokes / (stokes * (2.34 + 0.939 * (u0_u - 1)) + 1)
    return factor


def mean_free_path_air(P_pa, T_k):
    return 0.0664e-6 * (101325.0 / P_pa) * (T_k / 293.15) * ((1 + 110 / 293.15) / (1 + 110 / T_k))


def Kn(dp_m, P_pa, T_k):
    return 2.0 * mean_free_path_air(P_pa, T_k) / dp_m


def Cc(dp_m, P_pa, T_k):
    kn = Kn(dp_m, P_pa, T_k)
    return 1.0 + kn * (1.142 + 0.558 * np.exp(-0.999 / kn))


def tau_rel(rho_p, dp_m, mu, P_pa, T_k):
    return rho_p * dp_m**2 * Cc(dp_m, P_pa, T_k) / (18.0 * mu)


def n_sedim(d, L, rho_p, dp_m, mu, P_pa, T_k, Q):
    v_ts = tau_rel(rho_p, dp_m, mu, P_pa, T_k) * G
    return np.exp(-(d * L * v_ts) / Q)


def sherwood_number(re, sc, d, L):
    if re < 2300:
        x = re * sc * d / L
        return 3.66 + (0.0668 * x) / (1 + 0.04 * x ** (2 / 3))
    return 0.0118 * re ** (7 / 8) * sc ** (1 / 3)


def n_dif(dp_m, P_pa, T_k, d, L, Q, mu, re, rho_f):
    b = Cc(dp_m, P_pa, T_k) / (3 * pi * mu * dp_m)
    D = K_BOLTZMANN * T_k * b
    zeta = pi * D * L / Q
    sc = mu / (rho_f * D)
    sh = sherwood_number(re, sc, d, L)
    return np.exp(-zeta * sh)


def n_turb_inert(d, L, Q, re, V, dp_m, rho_p, mu):
    if re < 2300:
        return 1.0
    stokes = core.Stokes_number(V, dp_m, d, rho_p, mu)
    tau_plus = 0.0395 * stokes * re ** (3 / 4)
    v_plus = 0.1 if tau_plus > 12.9 else 6e-4 * tau_plus**2 + 2e-8 * re
    vt = v_plus * V / (5.03 * re ** (1 / 8))
    return np.exp(-(pi * d * L * vt) / Q)


def n_iner_codo(V, dp_m, d, rho_p, mu, P_pa, T_k, ang_deg, re=None):
    stokes = core.Stokes_number(V, dp_m, d, rho_p, mu) * Cc(dp_m, P_pa, T_k)
    if re is not None and re >= 2300:
        return exp(-2.823 * stokes * ang_deg * pi / 180)

    theta_kr = ang_deg * pi / 180.0
    term = (stokes / 0.171) ** (0.452 * (stokes / 0.171) + 2.242)
    return (1.0 + term) ** (-(2.0 / pi) * theta_kr)


def describe_flow_regime(re, threshold=2300):
    regime = "laminar" if re < threshold else "turbulento"
    comp = "<" if re < threshold else ">="
    return f"Regimen: {regime} porque Re = {re:.1f} {comp} {threshold}"


def _parse_decimal_field(value):
    text = str(value).strip().strip('"').replace(",", ".")
    if not text:
        raise ValueError("campo vacio")
    return float(text)


def _load_cfd_curve(base_dir: Path, filename_pattern: str, u0: float, label_template: str):
    """Helper genérico para cargar curvas CFD (muestreo o entrada)."""
    speed_tag = int(round(float(u0)))
    csv_path = base_dir / "CFD_JUAN" / filename_pattern.format(speed=speed_tag)
    print(f"[CFD] Buscando archivo: {csv_path}")

    if not csv_path.exists():
        print(f"[CFD] No se encontro {csv_path.name}; se omite.")
        return None

    dp_um = []
    eta = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x_val = _parse_decimal_field(row.get("x", ""))
                y_val = _parse_decimal_field(row.get("Curve1", ""))
            except (ValueError, TypeError):
                continue
            if x_val > 0:
                dp_um.append(x_val)
                eta.append(y_val)

    if not dp_um:
        print(f"[CFD] {csv_path.name} no contiene datos validos; se omite.")
        return None

    order = np.argsort(dp_um)
    print(f"[CFD] Usando archivo: {csv_path.name} ({len(dp_um)} puntos)")
    return {
        "dp_um": np.asarray(dp_um)[order],
        "eta": np.asarray(eta)[order],
        "label": label_template.format(speed=speed_tag),
    }


def load_cfd_sampling_curve(base_dir: Path, u0):
    return _load_cfd_curve(base_dir, "eff_{speed}ms.csv", u0, "CFD muestreo ({speed} m/s)")


def load_cfd_aspiration_curve(base_dir: Path, u0):
    return _load_cfd_curve(base_dir, "asp_{speed}.csv", u0, "CFD entrada ({speed} m/s)")


def print_entry_efficiency_parameters(U0, U1, U2, d1, d2, L1, L2, rho_p, atmos, dp_m):
    """Imprime parametros clave para auditar el calculo de eficiencias de entrada."""
    dp_arr = np.asarray(dp_m, dtype=float).reshape(-1)
    dp_min = float(np.min(dp_arr))
    dp_max = float(np.max(dp_arr))
    n_points = int(dp_arr.size)

    cc1_min = Cc(dp_min, atmos.P, atmos.T)
    cc1_max = Cc(dp_max, atmos.P, atmos.T)
    stk1_min = core.Stokes_number(U0, dp_min, d1, rho_p, atmos.mu) * cc1_min
    stk1_max = core.Stokes_number(U0, dp_max, d1, rho_p, atmos.mu) * cc1_max

    cc2_min = Cc(dp_min, atmos.P, atmos.T)
    cc2_max = Cc(dp_max, atmos.P, atmos.T)
    stk2_min = core.Stokes_number(U1, dp_min, d2, rho_p, atmos.mu) * cc2_min
    stk2_max = core.Stokes_number(U1, dp_max, d2, rho_p, atmos.mu) * cc2_max

    print("Parametros usados para eficiencia de entrada")
    print(f"- Rango dp: {dp_min*1e6:.3f} a {dp_max*1e6:.3f} um ({n_points} puntos)")
    print(f"- Propiedades: rho_p={rho_p:.1f} kg/m3, rho_f={atmos.rho:.4f} kg/m3, mu={atmos.mu:.3e} Pa s, P={atmos.P:.1f} Pa, T={atmos.T:.2f} K")
    print(f"- Seccion 1 (U0->U1): U0={U0:.3f} m/s, U1={U1:.3f} m/s, U0/U1={U0/U1:.4f}, d1={d1*1e3:.2f} mm, L1={L1*1e3:.2f} mm")
    print(f"  Cc(dp_min/max)={cc1_min:.4f}/{cc1_max:.4f}, Stk1(dp_min/max)={stk1_min:.4e}/{stk1_max:.4e}")
    print(f"- Seccion 2 (U1->U2): U1={U1:.3f} m/s, U2={U2:.3f} m/s, U1/U2={U1/U2:.4f}, d2={d2*1e3:.2f} mm, L2={L2*1e3:.2f} mm")
    print(f"  Cc(dp_min/max)={cc2_min:.4f}/{cc2_max:.4f}, Stk2(dp_min/max)={stk2_min:.4e}/{stk2_max:.4e}")
    print("  Ecuacion aplicada: eta_entrada = n_asp(ratio_velocidad, Stk) * n_trans(ratio_velocidad, Stk)")


def compute_shroud_probe(atmos, U0):
    # Parametros dados para sonda tipo envuelta
    rho_p = 1000.0

    # Seccion 1 (shroud)
    d1 = 20e-3
    L1 = 30e-3
    ang1 = 0.0
    U1 = 0.19 * U0
    Q1 = U1 * (pi * d1**2) / 4.0

    # Seccion 2 (linea interna de muestreo)
    d2 = 8e-3
    L2 = 250e-3
    ang2 = 0.0
    Q2 = 2.0 / 60000.0
    U2 = 4.0 * Q2 / (pi * d2**2)

    re1 = core.Reynolds(U1, d1, atmos.rho, atmos.mu)
    re2 = core.Reynolds(U2, d2, atmos.rho, atmos.mu)

    if not 8900 <= re2 <= 54000:
        print("[Gong] Aviso: el Reynolds del segundo tramo esta fuera del rango 8900-54000.")

    print("\n" + "=" * 70)
    print("Sonda tipo envuelta - parametros")
    print(f"U0 = {U0:.2f} m/s")
    print(f"Seccion 1: d1 = {d1*1e3:.1f} mm, L1 = {L1*1e3:.1f} mm, U1/U0 = {U1/U0:.3f}, Q1 = {Q1*60000:.2f} L/min")
    print(f"Seccion 2: d2 = {d2*1e3:.1f} mm, L2 = {L2*1e3:.1f} mm, Q2 = {Q2*60000:.2f} L/min, U2 = {U2:.3f} m/s")
    print(f"Seccion 1 -> Re1 = {re1:.1f} ({describe_flow_regime(re1)})")
    print(f"Seccion 2 -> Re2 = {re2:.1f} ({describe_flow_regime(re2)})")
    print("=" * 70 + "\n")

    dp_m = np.linspace(0.1, 50, 1000) * 1e-6
    dp_um = dp_m * 1e6
    dp_values = dp_m.tolist()

    print_entry_efficiency_parameters(U0, U1, U2, d1, d2, L1, L2, rho_p, atmos, dp_m)

    entrada1, entrada2, entrada1_corr, entrada2_corr, entrada_combinada, entrada_combinada_corr, transporte1, transporte2, factor_gong, total, total_corr = ([] for _ in range(11))

    for dp in dp_values:
        # Entrada seccion 1: flujo externo U0 -> flujo en shroud U1
        stk1_entry = core.Stokes_number(U0, dp, d1, rho_p, atmos.mu) * Cc(dp, atmos.P, atmos.T)
        ent1 = n_asp(U0 / U1, stk1_entry) * n_trans(U0 / U1, stk1_entry)

        # Perdidas transporte seccion 1
        sed1 = n_sedim(d1, L1, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q1)
        dif1 = n_dif(dp, atmos.P, atmos.T, d1, L1, Q1, atmos.mu, re1, atmos.rho)
        turb1 = n_turb_inert(d1, L1, Q1, re1, U1, dp, rho_p, atmos.mu)
        codo1 = n_iner_codo(U1, dp, d1, rho_p, atmos.mu, atmos.P, atmos.T, ang1, re1)
        tr1 = sed1 * dif1 * turb1 * codo1

        # Entrada seccion 2: flujo en shroud U1 -> flujo en linea interna U2
        stk2_entry = core.Stokes_number(U1, dp, d2, rho_p, atmos.mu) * Cc(dp, atmos.P, atmos.T)
        ent2 = n_asp(U1 / U2, stk2_entry) * n_trans(U1 / U2, stk2_entry)

        # Perdidas transporte seccion 2
        sed2 = n_sedim(d2, L2, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q2)
        dif2 = n_dif(dp, atmos.P, atmos.T, d2, L2, Q2, atmos.mu, re2, atmos.rho)
        turb2 = n_turb_inert(d2, L2, Q2, re2, U2, dp, rho_p, atmos.mu)
        codo2 = n_iner_codo(U2, dp, d2, rho_p, atmos.mu, atmos.P, atmos.T, ang2, re2)
        tr2 = sed2 * dif2 * turb2 * codo2

        # Factor de correccion de Gong et al. usando valores del segundo tramo.
        gong = factor_corr(U2 / U1, stk2_entry, re2)
        eta_total = ent1 * tr1 * ent2 * tr2
        eta_total_corr = eta_total * gong

        entrada1.append(ent1)
        entrada2.append(ent2)
        entrada1_corr.append(ent1 * gong)
        entrada2_corr.append(ent2 * gong)
        entrada_combinada.append(ent1 * ent2)
        entrada_combinada_corr.append(ent1 * ent2 * gong)
        transporte1.append(tr1)
        transporte2.append(tr2)
        factor_gong.append(gong)
        total.append(eta_total)
        total_corr.append(eta_total_corr)

    return {
        "U0": U0,
        "U1": U1,
        "U2": U2,
        "dp_um": dp_um,
        "entrada1": np.array(entrada1),
        "entrada2": np.array(entrada2),
        "entrada1_corr": np.array(entrada1_corr),
        "entrada2_corr": np.array(entrada2_corr),
        "entrada_combinada": np.array(entrada_combinada),
        "entrada_combinada_corr": np.array(entrada_combinada_corr),
        "total": np.array(total),
        "total_corr": np.array(total_corr),
        "re1": re1,
        "re2": re2,
    }


def _style_common_axes(ax):
    ax.set_xscale("log")
    ax.axhline(1.0, color="#808080", linewidth=1.2, alpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, which="both")


def _relative_difference_percent(x_model, model_values, x_cfd, cfd_values):
    x_model = np.asarray(x_model, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    x_cfd = np.asarray(x_cfd, dtype=float)
    cfd_values = np.asarray(cfd_values, dtype=float)
    cfd_interp = np.interp(x_model, x_cfd, cfd_values, left=np.nan, right=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(cfd_interp != 0, (model_values - cfd_interp) / cfd_interp * 100.0, np.nan)
    return rel


def plot_total_sampling_figure(base_dir: Path, cases):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    for case, color in zip(cases, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        ax_top.plot(case["model"]["dp_um"], case["model"]["total_corr"], color=color, linewidth=2.4, label=f"Modelo {case['U0']:.0f} m/s")
        if case["cfd_sampling"] is not None:
            ax_top.plot(
                case["cfd_sampling"]["dp_um"],
                case["cfd_sampling"]["eta"],
                color=color,
                linewidth=2.0,
                marker="x",
                markersize=5,
                markeredgewidth=1.2,
                markevery=3,
                label=f"CFD {case['U0']:.0f} m/s",
            )
            rel_diff = _relative_difference_percent(
                case["model"]["dp_um"],
                case["model"]["total_corr"],
                case["cfd_sampling"]["dp_um"],
                case["cfd_sampling"]["eta"],
            )
            rel_diff = np.where(case["model"]["dp_um"] <= DIFF_REL_PLOT_MAX_UM, rel_diff, np.nan)
            ax_bottom.plot(case["model"]["dp_um"], rel_diff, color=color, linewidth=2.4, label=f"{case['U0']:.0f} m/s")

    _style_common_axes(ax_top)
    _style_common_axes(ax_bottom)
    ax_bottom.axhline(0.0, color="#808080", linewidth=1.2, alpha=0.8)
    ax_bottom.set_ylim(-DIFF_REL_PLOT_YLIM_PERCENT, DIFF_REL_PLOT_YLIM_PERCENT)
    ax_top.set_ylabel("Eficiencia de muestreo total [-]", fontsize=18)
    ax_bottom.set_xlabel("Diametro de particula, $d_p$ [um]", fontsize=18)
    ax_bottom.set_ylabel("Dif. rel. [%]", fontsize=16)
    ax_top.legend(loc="best", frameon=True, fontsize=11, ncol=2)
    ax_bottom.legend(loc="best", frameon=True, fontsize=10, ncol=4)

    out_base = "eficiencia_sonda_envuelta_total_v2_shroud"
    fig.savefig(base_dir / f"{out_base}.png", bbox_inches="tight")
    fig.savefig(base_dir / f"{out_base}.pdf", bbox_inches="tight")
    print(f"✓ Graficos guardados: {out_base}.png y .pdf")


def plot_aspiration_subplots(base_dir: Path, cases):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, case in zip(axes, cases):
        color = "#1f77b4"
        ax.plot(case["model"]["dp_um"], case["model"]["entrada_combinada"], color=color, linewidth=2.4, label="Modelo")
        if case["cfd_aspiration"] is not None:
            ax.plot(
                case["cfd_aspiration"]["dp_um"],
                case["cfd_aspiration"]["eta"],
                color="#000000",
                linewidth=2.0,
                marker="x",
                markersize=5,
                markeredgewidth=1.2,
                markevery=3,
                label="CFD",
            )
        _style_common_axes(ax)
        ax.text(0.05, 0.92, f"U0 = {case['U0']:.0f} m/s", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("Diametro de particula, $d_p$ [um]", fontsize=14)
        ax.set_ylabel("Eficiencia de aspiracion [-]", fontsize=14)
        ax.legend(loc="best", frameon=True, fontsize=10)

    out_base = "eficiencia_sonda_envuelta_aspiracion_v2_shroud"
    fig.savefig(base_dir / f"{out_base}.png", bbox_inches="tight")
    fig.savefig(base_dir / f"{out_base}.pdf", bbox_inches="tight")
    print(f"✓ Graficos guardados: {out_base}.png y .pdf")


def _build_model_efficiency_table(cases, ranges_um):
    speeds = [int(round(case["U0"])) for case in cases]
    rows = []
    for low_um, high_um in ranges_um:
        mid_um = 0.5 * (low_um + high_um)
        row = [f"{low_um:g}-{high_um:g}"]
        for case in cases:
            eta = np.interp(mid_um, case["model"]["dp_um"], case["model"]["total_corr"])
            row.append(f"{eta:.4f}")
        rows.append(row)
    columns = ["Rango dp [um]"] + [f"U0={u} m/s" for u in speeds]
    return columns, rows


def export_model_efficiency_table_pdf(base_dir: Path, cases):
    columns, rows = _build_model_efficiency_table(cases, TABLE_PARTICLE_RANGES_UM)
    fig, ax = plt.subplots(1, 1, figsize=(11.7, 8.3), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8E8E8")

    out_name = "tabla_eficiencia_modelo_rangos_v2_shroud.pdf"
    fig.savefig(base_dir / out_name, bbox_inches="tight")
    print(f"✓ Tabla PDF guardada: {out_name}")


def main(show=True):
    base_dir = Path(__file__).resolve().parent
    atmos = ATMOSPHERE_1976(100)

    speeds = [8.0, 11.0, 14.0, 25.0]
    cases = []
    for u0 in speeds:
        model = compute_shroud_probe(atmos, u0)
        cfd_sampling = load_cfd_sampling_curve(base_dir, model["U0"])
        cfd_aspiration = load_cfd_aspiration_curve(base_dir, model["U0"])
        cases.append({"U0": u0, "model": model, "cfd_sampling": cfd_sampling, "cfd_aspiration": cfd_aspiration})

    plot_total_sampling_figure(base_dir, cases)
    plot_aspiration_subplots(base_dir, cases)
    export_model_efficiency_table_pdf(base_dir, cases)

    if show:
        plt.show()


if __name__ == "__main__":
    main(show=True)

