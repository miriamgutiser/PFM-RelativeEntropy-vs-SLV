"""
Compara los precios benchmark de entrada con los precios WMC calibrados.

La tabla resultante usa la misma malla activa de ``data_loader.py``:
filas de vencimientos, columnas de strikes y precios de referencia
``TARGET_OPTION_PRICES``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.types import OptionType
from Entropia_Relativa.data_loader import (
    CP,
    K,
    MATURITY_LABELS,
    T,
    TARGET_OPTION_PRICES,
    calibration_filename,
    prior_filename,
)
from Entropia_Relativa.wmc_precios import _load_paths_and_weights, _option_type, wmc_price


DISPLAY_DECIMALS = 7


def _module_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _option_type_name(option_type: OptionType) -> str:
    return "CALL" if option_type == OptionType.CALL else "PUT"


def build_comparison_rows(
    option_type: OptionType,
    prior_file: str,
    calibration_file: str,
) -> list[dict[str, float | str]]:
    path_matrix, times, weights = _load_paths_and_weights(
        prior_file=str(_module_path(prior_file)),
        calibration_file=str(_module_path(calibration_file)),
    )

    rows: list[dict[str, float | str]] = []
    for maturity_index, (maturity_label, maturity) in enumerate(zip(MATURITY_LABELS, T)):
        for strike_index, strike in enumerate(K):
            reference_price = float(TARGET_OPTION_PRICES[maturity_index, strike_index])
            wmc_reference_price = wmc_price(
                option_type=option_type,
                strike=float(strike),
                maturity=float(maturity),
                path_matrix=path_matrix,
                times=times,
                weights=weights,
            )
            difference = wmc_reference_price - reference_price
            rows.append(
                {
                    "tipo": _option_type_name(option_type),
                    "vencimiento": str(maturity_label),
                    "T": float(maturity),
                    "strike": float(strike),
                    "referencia": reference_price,
                    "wmc": wmc_reference_price,
                    "diferencia": difference,
                    "error_absoluto": abs(difference),
                }
            )
    return rows


def _format_float(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def format_text_table(rows: list[dict[str, float | str]], decimals: int) -> str:
    header = (
        f"{'Tipo':<6} {'Venc.':>6} {'T':>10} {'Strike':>10} "
        f"{'Referencia':>14} {'WMC':>14} {'WMC-Ref':>14} {'|Error|':>14}"
    )
    lines = [header, "-" * len(header)]
    current_maturity = None

    for row in rows:
        if current_maturity is not None and row["vencimiento"] != current_maturity:
            lines.append("")
        current_maturity = row["vencimiento"]

        lines.append(
            f"{str(row['tipo']):<6} "
            f"{str(row['vencimiento']):>6} "
            f"{float(row['T']):>10.6f} "
            f"{float(row['strike']):>10.2f} "
            f"{float(row['referencia']):>14.{decimals}f} "
            f"{float(row['wmc']):>14.{decimals}f} "
            f"{float(row['diferencia']):>14.{decimals}f} "
            f"{float(row['error_absoluto']):>14.{decimals}f}"
        )

    max_abs_error = max(float(row["error_absoluto"]) for row in rows)
    mae = float(np.mean([float(row["error_absoluto"]) for row in rows]))
    lines.extend(
        [
            "",
            f"Max |WMC - referencia| = {max_abs_error:.{decimals}f}",
            f"MAE = {mae:.{decimals}f}",
        ]
    )
    return "\n".join(lines)


def format_latex_table(rows: list[dict[str, float | str]], decimals: int) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{ccrrrr}",
        r"\hline",
        r"Vencimiento & Strike & Precio referencia & Precio WMC & WMC-Ref. & Error absoluto \\",
        r"\hline",
    ]

    for row in rows:
        lines.append(
            f"{row['vencimiento']} & "
            f"{_format_float(float(row['strike']), 2)} & "
            f"{_format_float(float(row['referencia']), decimals)} & "
            f"{_format_float(float(row['wmc']), decimals)} & "
            f"{_format_float(float(row['diferencia']), decimals)} & "
            f"{_format_float(float(row['error_absoluto']), decimals)} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Comparación entre precios de referencia y precios WMC.}",
            r"\label{tab:wmc_benchmark_comparison}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara TARGET_OPTION_PRICES con los precios WMC calculados "
            "sobre la misma malla de strikes y vencimientos."
        )
    )
    parser.add_argument(
        "--option-type",
        type=_option_type,
        default=CP,
        help="Tipo de opcion usado para valorar por WMC: call o put.",
    )
    parser.add_argument("--prior-file", default=prior_filename)
    parser.add_argument("--calibration-file", default=calibration_filename)
    parser.add_argument(
        "--format",
        choices=["text", "latex"],
        default="text",
        help="Formato de salida de la tabla.",
    )
    parser.add_argument("--decimals", type=int, default=DISPLAY_DECIMALS)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta opcional donde guardar la tabla generada.",
    )
    return parser


def main() -> None:
    args, _ = build_parser().parse_known_args()
    try:
        rows = build_comparison_rows(
            option_type=args.option_type,
            prior_file=args.prior_file,
            calibration_file=args.calibration_file,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Error cargando los datos WMC: {exc}\n"
            "Regenera la simulacion prior y la calibracion con el mismo "
            "numero de trayectorias antes de construir esta tabla."
        ) from exc

    if args.format == "latex":
        table = format_latex_table(rows=rows, decimals=args.decimals)
    else:
        table = format_text_table(rows=rows, decimals=args.decimals)

    if args.output is not None:
        Path(args.output).write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
