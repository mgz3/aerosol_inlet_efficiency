from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from matplotlib import pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "New_PLC"


@dataclass(frozen=True)
class ProfileSpec:
    u0: int
    filename: str
    color: str


PROFILE_SPECS = [
    ProfileSpec(u0=25, filename="perfil_vel_25ms.csv", color="#d62728"),  # rojo
    ProfileSpec(u0=14, filename="perfil_vel_14ms.csv", color="#2ca02c"),  # verde
    ProfileSpec(u0=11, filename="perfil_vel_11ms.csv", color="#e6b800"),  # amarillo
    ProfileSpec(u0=8, filename="perfil_vel_8ms.csv", color="#1f77b4"),    # azul
]


def _to_float(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _load_profile(path: Path):
    df = pd.read_csv(path)
    if "x" not in df.columns:
        raise ValueError(f"No se encontro la columna 'x' en {path.name}")

    y_col = next((col for col in df.columns if col.lower().startswith("vel_")), None)
    if y_col is None:
        y_col = next((col for col in df.columns if col != "x"), None)
    if y_col is None:
        raise ValueError(f"No se encontro columna de velocidad en {path.name}")

    x = _to_float(df["x"])
    y = _to_float(df[y_col])

    clean = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if clean.empty:
        raise ValueError(f"{path.name} no tiene datos validos")

    return clean["x"].to_numpy(), clean["y"].to_numpy()


def _configure_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        # Fallback si scienceplots no esta instalado.
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.size": 13,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.8,
            "font.family": "serif",
        }
    )


def plot_velocity_profiles(show=False):
    _configure_style()

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)

    for spec in PROFILE_SPECS:
        path = DATA_DIR / spec.filename
        x_mm, u_u0 = _load_profile(path)
        ax.plot(
            x_mm,
            u_u0,
            color=spec.color,
            label=f"U0 = {spec.u0} m/s",
        )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("U/U0")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="best", frameon=True)

    out_base = BASE_DIR / "perfil_vel_CFD"
    fig.savefig(f"{out_base}.png", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print("✓ Graficos guardados: perfil_vel_CFD.png y perfil_vel_CFD.pdf")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_velocity_profiles(show=False)

