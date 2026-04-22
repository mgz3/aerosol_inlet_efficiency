from math import exp, pi
from pathlib import Path

import numpy as np
import pandas as pd
from fluids import ATMOSPHERE_1976, core
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import scienceplots


plt.style.use(["science", "no-latex"])
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.4,
        "axes.linewidth": 1.2,
        "font.family": "serif",
    }
)

K_BOLTZMANN = 1.380649e-23
G = 9.81
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


def _to_float_series(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_plc_data(base_dir: Path):
    plc_dir = base_dir / "New_PLC"
    files = {
        "asp": "asp.csv",
        "diff": "difusion.csv",
        "muestreo": "muestreo.csv",
        "sed": "sed.csv",
        "codo": "codo.csv",
        "transport_loss": "transport_loss.csv",
    }
    y_aliases = {
        "asp": ["asp", "curve1"],
        "diff": ["diff", "curve1"],
        "muestreo": ["muestreo", "sampling", "curve1"],
        "sed": ["sed", "sedim", "curve1"],
        "codo": ["codo", "curve1"],
        "transport_loss": ["transport_loss", "curve1"],
    }

    data = {}
    for key, filename in files.items():
        path = plc_dir / filename
        if not path.exists():
            print(f"[PLC] Aviso: no existe {path.name}, se omite.")
            continue

        df = pd.read_csv(path)
        col_map = {col.strip().lower(): col for col in df.columns}

        x_col = col_map.get("x")
        if x_col is None:
            print(f"[PLC] Aviso: no se encontro columna x en {path.name}, se omite.")
            continue

        y_col = next((col_map[name] for name in y_aliases[key] if name in col_map), None)
        if y_col is None:
            y_col = next((col for col in df.columns if col != x_col), None)
        if y_col is None:
            print(f"[PLC] Aviso: no se encontro columna Y en {path.name}, se omite.")
            continue

        x = _to_float_series(df[x_col])
        y = _to_float_series(df[y_col])
        clean = pd.DataFrame({"x": x, "y": y}).dropna()
        clean = clean[clean["x"] > 0].sort_values("x")

        if clean.empty:
            print(f"[PLC] Aviso: {path.name} no tiene datos validos para graficar.")
            continue

        data[key] = {"x": clean["x"].to_numpy(), "y": clean["y"].to_numpy()}

    return data


def n_asp(u0_u, stokes):
    return 1 + (u0_u - 1) * (1 - (1 + stokes * (2 + 0.617 / u0_u)) ** -1)


def n_trans(u0_u, stokes):
    return (1 + (u0_u - 1) / (1 + 2.66 / stokes ** (2 / 3))) / (
        1 + (u0_u - 1) / (1 + 0.418 / stokes)
    )


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

    # Si no se proporciona Reynolds, se conserva la correlacion laminar actual.
    # Umbral clasico de transicion laminar/turbulento.
    if re is not None and re >= 2300:
        # Formula comentada originalmente en el script para flujo turbulento.
        return exp(-2.823 * stokes * ang_deg * pi / 180)

    # Correlacion bibliografica para flujo laminar en codo.
    theta_kr = ang_deg * pi / 180.0
    term = (stokes / 0.171) ** (0.452 * (stokes / 0.171) + 2.242)
    eta = (1.0 + term) ** (-(2.0 / pi) * theta_kr)
    return eta


def describe_flow_regime(re, threshold=2300):
    if re < threshold:
        return (
            f"Regimen en el codo: laminar porque Re = {re:.1f} < {threshold}; "
            "se usa la correlacion bibliografica laminar implementada en n_iner_codo()."
        )
    return (
        f"Regimen en el codo: turbulento porque Re = {re:.1f} >= {threshold}; "
        "se usa la formula comentada exp(-2.823 * Stk * angulo)."
    )


def describe_reynolds_calculation(Q, d, U, rho, mu, re):
    area = pi * d**2 / 4.0
    return (
        "Calculo de Reynolds:\n"
        f"  Re = rho * U * d / mu\n"
        f"  Q   = {Q:.6e} m3/s ({Q * 60000:.3f} L/min)\n"
        f"  d   = {d:.6e} m\n"
        f"  A   = pi * d^2 / 4 = {area:.6e} m2\n"
        f"  U   = 4 * Q / (pi * d^2) = {U:.6f} m/s\n"
        f"  rho = {rho:.6f} kg/m3\n"
        f"  mu  = {mu:.6e} Pa·s\n"
        f"  Re  = {re:.1f}"
    )


def compute_two_stage_curved_probe(atmos, U0=15, verbose=True):
    # Geometria pedida (coincide con PLC)
    d = 8e-3           # [m] diameter = 8 mm
    L1 = 25e-3         # [m] Seccion 1: L1 = 25 mm
    L2 = 10e-3         # [m] Seccion 2: L2 = 10 mm
    L_rect = L1 + L2   # [m] Longitud total = 35 mm
    ang_sec1 = 0       # [deg] Primera seccion recta
    ang_sec2 = 60      # [deg] Segunda seccion curvada

    # Flujo y entorno (coincide con PLC)
    Q = 2 / 60000      # [m3/s] Q = 2 L/min
    U = 4 * Q / (pi * d**2)
    re = core.Reynolds(U, d, atmos.rho, atmos.mu)
    if verbose:
        print(describe_reynolds_calculation(Q, d, U, atmos.rho, atmos.mu, re))

    rho_p = 1000
    dp_m = np.linspace(0.1, 50, 1000) * 1e-6
    dp_um = dp_m * 1e6
    u0_u = U0 / U

    asp, trans, entrada, sedim, difus, turb_inert, codo, transport_total, muestreo = ([] for _ in range(9))

    for dp in dp_m:
        stk = core.Stokes_number(U0, dp, d, rho_p, atmos.mu) * Cc(dp, atmos.P, atmos.T)

        # Componentes de muestreo
        asp_eff = n_asp(u0_u, stk)
        trans_eff = n_trans(u0_u, stk)
        entrada_eff = asp_eff * trans_eff

        # Componentes del tramo recto
        sedim_eff = n_sedim(d, L_rect, rho_p, dp, atmos.mu, atmos.P, atmos.T, Q)
        difus_eff = n_dif(dp, atmos.P, atmos.T, d, L_rect, Q, atmos.mu, re, atmos.rho)
        turb_eff = n_turb_inert(d, L_rect, Q, re, U, dp, rho_p, atmos.mu)

        # Codo
        codo_eff = n_iner_codo(U, dp, d, rho_p, atmos.mu, atmos.P, atmos.T, ang_sec2, re)

        # Transporte total interno (sin aspiracion ni transmision)
        transport_total_eff = sedim_eff * difus_eff * turb_eff * codo_eff

        # Eficiencia total de muestreo
        muestreo_eff = entrada_eff * transport_total_eff

        asp.append(asp_eff)
        trans.append(trans_eff)
        entrada.append(entrada_eff)
        sedim.append(sedim_eff)
        difus.append(difus_eff)
        turb_inert.append(turb_eff)
        codo.append(codo_eff)
        transport_total.append(transport_total_eff)
        muestreo.append(muestreo_eff)

    return {
        "dp_um": dp_um,
        "asp": np.array(asp),
        "trans": np.array(trans),
        "entrada": np.array(entrada),
        "sedim": np.array(sedim),
        "difus": np.array(difus),
        "turb_inert": np.array(turb_inert),
        "codo": np.array(codo),
        "transport_total": np.array(transport_total),
        "muestreo": np.array(muestreo),
        "re": re,
    }


def _build_model_efficiency_table(cases, ranges_um):
    speeds = [int(round(case["U0"])) for case in cases]
    rows = []
    for low_um, high_um in ranges_um:
        mid_um = 0.5 * (low_um + high_um)
        row = [f"{low_um:g}-{high_um:g}"]
        for case in cases:
            eta = np.interp(mid_um, case["model"]["dp_um"], case["model"]["muestreo"])
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

    out_name = "tabla_eficiencia_modelo_rangos_v2.pdf"
    fig.savefig(base_dir / out_name, bbox_inches="tight")
    print(f"✓ Tabla PDF guardada: {out_name}")


def _plot_efficiency_group(base_dir: Path, model, plc, compare_specs, title, out_base):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), constrained_layout=True)

    y_max = []
    model_lines = {}
    plc_lines = {}

    for spec in compare_specs:
        name = spec["name"]
        model_key = spec["model_key"]
        plc_key = spec["plc_key"]
        color = spec["color"]
        plc_to_eff = spec["plc_to_eff"]

        if model_key in model:
            y_model = model[model_key]
            (line_model,) = ax.plot(
                model["dp_um"],
                y_model,
                label=f"{name} - Modelo",
                color=color,
                linewidth=2.8,
                linestyle="-",
                zorder=2,
            )
            model_lines[name] = line_model
            y_max.append(np.nanmax(y_model))

        if plc and plc_key in plc:
            y_plc = plc_to_eff(plc[plc_key]["y"])
            (line_plc,) = ax.plot(
                plc[plc_key]["x"],
                y_plc,
                label=f"{name} - PLC",
                color=color,
                linewidth=3.0,
                linestyle=(0, (8, 4)),
                marker="x",
                markersize=10,
                markerfacecolor="white",
                markeredgewidth=0.9,
                markevery=3,
                alpha=1.0,
                zorder=3,
            )
            line_plc.set_path_effects([pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()])
            plc_lines[name] = line_plc
            y_max.append(np.nanmax(y_plc))

    ax.set_xlabel("Diametro de particula, $d_p$ [um]", fontsize=20)
    ax.set_ylabel("Eficiencia [-]", fontsize=20)
    ax.set_title(title, fontsize=18)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35, which="both")

    # Leyenda tabular en dos columnas: Modelo | PLC.
    # Matplotlib llena por columnas, por eso se arma primero toda la columna Modelo
    # y luego toda la columna PLC para que las filas queden alineadas por fenomeno.
    header_model = Line2D([], [], linestyle="None")
    header_plc = Line2D([], [], linestyle="None")
    empty_cell = Line2D([], [], linestyle="None")

    model_handles_col = [header_model]
    model_labels_col = [r"$\bf{Modelo}$"]
    plc_handles_col = [header_plc]
    plc_labels_col = [r"$\bf{PLC}$"]

    for spec in compare_specs:
        name = spec["name"]
        model_handles_col.append(model_lines.get(name, empty_cell))
        model_labels_col.append(name if name in model_lines else " ")
        plc_handles_col.append(plc_lines.get(name, empty_cell))
        plc_labels_col.append(name if name in plc_lines else " ")

    legend_handles = model_handles_col + plc_handles_col
    legend_labels = model_labels_col + plc_labels_col

    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        frameon=True,
        fontsize=12,
        ncol=2,
        handlelength=2.6,
        columnspacing=1.6,
        handletextpad=0.8,
    )
    if y_max:
        ax.set_ylim(0, max(y_max) * 1.08)

    fig.savefig(base_dir / f"{out_base}.png", bbox_inches="tight")
    fig.savefig(base_dir / f"{out_base}.pdf", bbox_inches="tight")
    print(f"✓ Graficos guardados: {out_base}.png y .pdf")


def plot_probe_efficiency(base_dir: Path, model, plc=None):
    # Figura 1: comparacion de entrada y eficiencia total de muestreo
    compare_specs_entrada_total = [
        {"name": "Entrada", "model_key": "entrada", "plc_key": "asp", "plc_to_eff": lambda y: y, "color": "#1f77b4"},
        {"name": "Muestreo total", "model_key": "muestreo", "plc_key": "muestreo", "plc_to_eff": lambda y: y, "color": "#8c564b"},
    ]
    _plot_efficiency_group(
        base_dir,
        model,
        plc,
        compare_specs_entrada_total,
        "Comparacion de eficiencia de entrada y total (Modelo vs PLC New)",
        "comparacion_entrada_total_v2",
    )

    # Figura 2: todas las eficiencias de transporte
    compare_specs_transporte = [
        {"name": "Difusion", "model_key": "difus", "plc_key": "diff", "plc_to_eff": lambda y: 1 - y, "color": "#ff7f0e"},
        {"name": "Sedimentacion", "model_key": "sedim", "plc_key": "sed", "plc_to_eff": lambda y: 1 - y, "color": "#2ca02c"},
        {"name": "Codo", "model_key": "codo", "plc_key": "codo", "plc_to_eff": lambda y: 1 - y, "color": "#d62728"},
        {
            "name": "Transporte total",
            "model_key": "transport_total",
            "plc_key": "transport_loss",
            "plc_to_eff": lambda y: 1 - y,
            "color": "#9467bd",
        },
    ]
    _plot_efficiency_group(
        base_dir,
        model,
        plc,
        compare_specs_transporte,
        "Comparacion de eficiencias de transporte (Modelo vs PLC New)",
        "comparacion_eficiencias_transporte_v2",
    )


def plot_relative_difference_muestreo(base_dir: Path, model, plc=None):
    if not plc or "muestreo" not in plc:
        print("[PLC] Aviso: no se encontro 'muestreo.csv'; no se genera el grafico de diferencia relativa.")
        return

    x_model = model["dp_um"]
    y_model = model["muestreo"]
    x_plc = plc["muestreo"]["x"]
    y_plc = plc["muestreo"]["y"]

    x_min = max(np.min(x_model), np.min(x_plc))
    x_max = min(np.max(x_model), np.max(x_plc))
    mask = (x_model >= x_min) & (x_model <= x_max)

    x_common = x_model[mask]
    y_model_common = y_model[mask]
    y_plc_interp = np.interp(x_common, x_plc, y_plc)

    valid = np.abs(y_model_common) > 1e-12
    x_common = x_common[valid]
    y_model_common = y_model_common[valid]
    y_plc_interp = y_plc_interp[valid]

    rel_diff_pct = ((y_plc_interp - y_model_common) / y_model_common) * 100.0

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
    ax.plot(x_common, rel_diff_pct, color="#8A2BE2", linewidth=2.4, label="(PLC - Modelo) / Modelo")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.6, alpha=0.7)
    # ax.set_xscale("log")
    # ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Diámetro de partícula, $d_p$ [µm]", fontsize=16)
    ax.set_ylabel("Diferencia relativa [%]", fontsize=16)
    ax.set_title("Diferencia relativa porcentual de muestreo (PLC vs Modelo)", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30, which="both")
    ax.legend(loc="best", frameon=True, fontsize=9)

    out_base = "diferencia_relativa_muestreo_v2"
    fig.savefig(base_dir / f"{out_base}.png", bbox_inches="tight")
    fig.savefig(base_dir / f"{out_base}.pdf", bbox_inches="tight")
    print(f"✓ Grafico de diferencia relativa guardado: {out_base}.png y .pdf")


def main(show=True):
    base_dir = Path(__file__).resolve().parent
    atmos = ATMOSPHERE_1976(100)
    model = compute_two_stage_curved_probe(atmos)
    plc = load_plc_data(base_dir)

    print(f"\n{'='*70}")
    print(f"Reynolds en la sonda (d=8 mm): {model['re']:.1f}")
    print(describe_flow_regime(model["re"]))
    print(f"{'='*70}\n")

    # Analisis de discrepancia para comparacion en dos figuras
    print("[ANALISIS] Comparacion de eficiencias Modelo vs PLC New en dos graficos:")
    print("-" * 70)
    print("Grafico 1: entrada y muestreo total")
    print("Grafico 2: difusion, sedimentacion, codo y transporte total")
    print("  - Geometria: primera seccion 0° y segunda seccion 60°\n")
    print("Convencion PLC: asp/muestreo directas; diff/sed/codo/transport_loss convertidas como 1 - perdida\n")
    print("Referencia PLC:")
    print("  - Archivos en 'New_PLC/*.csv'\n")
    print("Causas posibles de discrepancia:")
    print("  x Diferencias en correlaciones de cada fenomeno")
    print("  x Uso de Cc, P, T, propiedades del aire o densidad de particula distintos")
    print("  x Diferencias en digitalizacion/normalizacion de los datos PLC\n")

    plot_probe_efficiency(base_dir, model, plc=plc)
    # plot_relative_difference_muestreo(base_dir, model, plc=plc)

    table_speeds = [8.0, 11.0, 14.0, 25.0]
    table_cases = [{"U0": u0, "model": compute_two_stage_curved_probe(atmos, U0=u0, verbose=False)} for u0 in table_speeds]
    export_model_efficiency_table_pdf(base_dir, table_cases)

    print(f"{'='*70}\n")

    if show:
        # plt.ion()  # Modo interactivo
        plt.show()


if __name__ == "__main__":
    main(show=True)

