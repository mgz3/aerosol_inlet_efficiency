"""
Análisis detallado de las discrepancias entre Muestreo Modelo vs PLC.
Permite identificar qué componente (asp, trans, Stokes, etc.) causa la diferencia.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from math import pi, exp
from fluids import ATMOSPHERE_1976, core

# Importar funciones del script principal
from validation_eff_method_with_PLC_v2 import (
    load_plc_data,
    n_asp,
    n_trans,
    Cc,
    _to_float_series,
)

def analyze_muestreo_discrepancy():
    base_dir = Path(__file__).resolve().parent
    atmos = ATMOSPHERE_1976(100)

    # Parámetros
    d = 8e-3
    Q = 2 / 60000
    U0 = 15
    U = 4 * Q / (pi * d**2)
    rho_p = 1000
    u0_u = U0 / U

    # Cargar datos PLC
    plc = load_plc_data(base_dir)

    print("\n" + "="*90)
    print("ANÁLISIS DE DISCREPANCIA: MUESTREO MODELO vs PLC")
    print("="*90)

    if "muestreo" not in plc or "asp" not in plc:
        print("❌ No se encontraron datos PLC de muestreo o aspiración.")
        return

    dp_plc = plc["asp"]["x"]  # Usar diámetros del PLC
    asp_plc = plc["asp"]["y"]
    muestreo_plc = plc["muestreo"]["y"]

    print(f"\n📊 Comparación en {len(dp_plc)} puntos de diámetro PLC:")
    print("-" * 90)
    print(f"{'dp [µm]':<12} {'asp_PLC':<12} {'asp_modelo':<12} {'trans_mod':<12} {'muest_PLC':<12} {'muest_mod':<12} {'ratio':<10}")
    print("-" * 90)

    ratios = []
    asp_errors = []

    for i, dp_um in enumerate(dp_plc):
        if i % (len(dp_plc) // 10) == 0 or i == len(dp_plc) - 1:  # Mostrar cada 10%
            dp_m = dp_um * 1e-6
            stk = core.Stokes_number(U0, dp_m, d, rho_p, atmos.mu) * Cc(dp_m, atmos.P, atmos.T)

            asp_model = n_asp(u0_u, stk)
            trans_model = n_trans(u0_u, stk)
            muestreo_model = asp_model * trans_model

            muestreo_plc_i = muestreo_plc[i]
            asp_plc_i = asp_plc[i]

            ratio = muestreo_plc_i / muestreo_model if muestreo_model > 0 else 0
            ratios.append(ratio)

            asp_error = abs(asp_model - asp_plc_i) / max(asp_plc_i, 1e-6) * 100
            asp_errors.append(asp_error)

            print(f"{dp_um:<12.2f} {asp_plc_i:<12.4f} {asp_model:<12.4f} {trans_model:<12.4f} {muestreo_plc_i:<12.4f} {muestreo_model:<12.4f} {ratio:<10.3f}")

    print("-" * 90)
    print(f"\n📈 Estadísticas:")
    print(f"  • Ratio medio (muestreo_PLC / muestreo_modelo): {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    print(f"  • Error aspiración (modelo vs PLC): {np.mean(asp_errors):.1f}% ± {np.std(asp_errors):.1f}%")
    print(f"  • Rango de ratios: {np.min(ratios):.3f} a {np.max(ratios):.3f}")

    print("\n🔍 Interpretación:")
    ratio_mean = np.mean(ratios)
    if abs(ratio_mean - 1.0) < 0.1:
        print("  ✓ Los valores están CERCANOS (ratio ~ 1.0)")
        print("    → Posibles causas: pequeñas diferencias en fórmulas empíricas")
    elif ratio_mean > 1.2:
        print(f"  ⚠ PLC muestreo >> Modelo (ratio={ratio_mean:.2f})")
        print("    → PLC captura MÁS partículas en muestreo")
        print("    → Verificar: aspiration_angle, fórmulas n_asp/n_trans, definición Stokes")
    elif ratio_mean < 0.8:
        print(f"  ⚠ PLC muestreo << Modelo (ratio={ratio_mean:.2f})")
        print("    → PLC captura MENOS partículas en muestreo")
        print("    → Verificar: definiciones de eficiencia, normalización de datos PLC")

    print("\n💡 Posibles causas de discrepancia:")
    print("  1. Definición diferente de aspiración:")
    print("     • Modelo usa: n_asp(u0_u, Stk) con fórmula empírica")
    print("     • PLC puede usar: digitalizaciones gráficas o método CFD distinto")
    print("  2. Cálculo de número de Stokes:")
    print("     • Diferencias en factor de Cunningham (Cc)")
    print("     • Densidad de partícula distinta")
    print("  3. Número de Richardson o ecuaciones de transporte:")
    print("     • Correlaciones empíricas (Sherwood, etc.)")
    print("  4. Influencia de aspiration_angle = 0:")
    print("     • Con ángulo = 0, hay máxima aspiración teórica")
    print("     • Verificar si PLC realmente usó angle = 0")

    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    analyze_muestreo_discrepancy()

