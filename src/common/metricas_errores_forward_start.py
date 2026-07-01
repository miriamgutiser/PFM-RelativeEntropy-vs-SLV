"""
Calcula MAE y error maximo entre precios forward-start WMC y SLV.

Uso rapido:
1. Pega la salida de wmc_forward_start.py en SERIE_A_TEXT.
2. Pega la salida de slv_forward_start_precios.py en SERIE_B_TEXT.
3. Ejecuta:

   python src/common/metricas_errores_forward_start.py

Tambien se pueden pasar ficheros de texto con --serie-a-file y --serie-b-file.
El parser usa solo filas que empiezan por CALL o PUT. Admite tanto la tabla
actual sin S0:

CALL        0.50       5.00        70.00          12.34

como una tabla antigua con S0:

CALL      100.00       0.50       5.00        70.00          12.34
"""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Pega aqui las tablas que imprimen los scripts, si quieres ejecutar sin CLI.
SERIE_A_NAME = "WMC"
SERIE_B_NAME = "SLV"
REQUIRE_SAME_GRID = False

SERIE_A_TEXT = """
Tipo           T1         T2            K     Precio WMC
--------------------------------------------------------
CALL         0.50       1.00        70.00          30.00
CALL         0.50       1.00        90.00          10.02
CALL         0.50       1.00       110.00           0.08

CALL         0.50       2.00        70.00          30.00
CALL         0.50       2.00        90.00          10.34
CALL         0.50       2.00       110.00           0.42

CALL         0.50       5.00        70.00          30.03
CALL         0.50       5.00        90.00          11.94
CALL         0.50       5.00       110.00           2.07

CALL         0.50      10.00        70.00          30.50
CALL         0.50      10.00        90.00          14.54
CALL         0.50      10.00       110.00           5.13

CALL         1.00       2.00        70.00          30.00
CALL         1.00       2.00        90.00          10.12
CALL         1.00       2.00       110.00           0.16

CALL         1.00       5.00        70.00          30.02
CALL         1.00       5.00        90.00          11.67
CALL         1.00       5.00       110.00           1.75

CALL         1.00      10.00        70.00          30.43
CALL         1.00      10.00        90.00          14.36
CALL         1.00      10.00       110.00           4.89

CALL         2.00       5.00        70.00          30.00
CALL         2.00       5.00        90.00          11.26
CALL         2.00       5.00       110.00           1.48

CALL         2.00      10.00        70.00          30.36
CALL         2.00      10.00        90.00          14.04
CALL         2.00      10.00       110.00           4.66

CALL         5.00      10.00        70.00          30.11
CALL         5.00      10.00        90.00          12.74
CALL         5.00      10.00       110.00           3.21
"""

SERIE_B_TEXT = """
Tipo           T1         T2            K     Precio SLV
--------------------------------------------------------------------
CALL         0.50       1.00        70.00          30.05
CALL         0.50       1.00        90.00          10.16
CALL         0.50       1.00       110.00           0.12

CALL         0.50       2.00        70.00          30.09
CALL         0.50       2.00        90.00          10.71
CALL         0.50       2.00       110.00           0.29

CALL         0.50       5.00        70.00          30.48
CALL         0.50       5.00        90.00          12.49
CALL         0.50       5.00       110.00           2.09

CALL         0.50      10.00        70.00          31.20
CALL         0.50      10.00        90.00          14.90
CALL         0.50      10.00       110.00           5.06

CALL         1.00       2.00        70.00          30.09
CALL         1.00       2.00        90.00          10.50
CALL         1.00       2.00       110.00           0.12

CALL         1.00       5.00        70.00          30.45
CALL         1.00       5.00        90.00          12.33
CALL         1.00       5.00       110.00           1.84

CALL         1.00      10.00        70.00          31.16
CALL         1.00      10.00        90.00          14.77
CALL         1.00      10.00       110.00           4.90

CALL         2.00       5.00        70.00          30.27
CALL         2.00       5.00        90.00          11.82
CALL         2.00       5.00       110.00           1.29

CALL         2.00      10.00        70.00          30.95
CALL         2.00      10.00        90.00          14.39
CALL         2.00      10.00       110.00           4.52

CALL         5.00      10.00        70.00          30.46
CALL         5.00      10.00        90.00          13.13
CALL         5.00      10.00       110.00           3.03
"""


# Alternativa: rellena vectores en orden par (T1,T2) -> strike.
OPTION_TYPE = "CALL"
T1_T2_PAIRS: tuple[tuple[float, float], ...] = ()
STRIKES: tuple[float, ...] = ()
SERIE_A_PRICES: tuple[float, ...] | tuple[tuple[float, ...], ...] = ()
SERIE_B_PRICES: tuple[float, ...] | tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class ForwardStartPoint:
    option_type: str
    t1: float
    t2: float
    strike: float
    price: float

    @property
    def key(self) -> tuple[str, float, float, float]:
        return (
            self.option_type.upper(),
            round(self.t1, 10),
            round(self.t2, 10),
            round(self.strike, 10),
        )


@dataclass(frozen=True)
class ErrorRow:
    option_type: str
    t1: float
    t2: float
    strike: float
    price_a: float
    price_b: float
    abs_error: float


def _parse_float(value: str) -> float:
    return float(value.strip().replace(",", "."))


def parse_forward_start_text(text: str) -> list[ForwardStartPoint]:
    """
    Extrae filas forward-start tipo ``CALL T1 T2 K precio``.

    Si la fila incluye S0 antes de T1, lo ignora automaticamente.
    """
    points: list[ForwardStartPoint] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().replace("|", " ")
        if not line:
            continue

        parts = line.split()
        option_type = parts[0].upper() if parts else ""
        if option_type not in {"CALL", "PUT"}:
            continue

        numeric_values: list[float] = []
        for part in parts[1:]:
            try:
                numeric_values.append(_parse_float(part))
            except ValueError:
                continue

        if len(numeric_values) < 4:
            continue

        t1, t2, strike, price = numeric_values[-4:]
        points.append(
            ForwardStartPoint(
                option_type=option_type,
                t1=t1,
                t2=t2,
                strike=strike,
                price=price,
            )
        )
    return points


def _flatten_price_values(values: Iterable) -> list[float]:
    flattened: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(float(item) for item in value)
        else:
            flattened.append(float(value))
    return flattened


def build_points_from_vectors(
    option_type: str,
    t1_t2_pairs: Iterable[tuple[float, float]],
    strikes: Iterable[float],
    prices: Iterable,
) -> list[ForwardStartPoint]:
    """
    Construye puntos en orden: para cada par (T1,T2), todos los strikes.
    """
    pair_values = [(float(t1), float(t2)) for t1, t2 in t1_t2_pairs]
    strike_values = [float(strike) for strike in strikes]
    price_values = _flatten_price_values(prices)
    expected = len(pair_values) * len(strike_values)
    if len(price_values) != expected:
        raise ValueError(
            "La serie de precios debe tener len(T1_T2_PAIRS) * len(STRIKES) valores. "
            f"Esperado: {expected}; recibido: {len(price_values)}."
        )

    normalized_type = option_type.strip().upper()
    if normalized_type not in {"CALL", "PUT"}:
        raise ValueError("option_type debe ser CALL o PUT.")

    points: list[ForwardStartPoint] = []
    index = 0
    for t1, t2 in pair_values:
        for strike in strike_values:
            points.append(
                ForwardStartPoint(
                    option_type=normalized_type,
                    t1=t1,
                    t2=t2,
                    strike=strike,
                    price=float(price_values[index]),
                )
            )
            index += 1
    return points


def _points_by_key(
    points: list[ForwardStartPoint],
    name: str,
) -> dict[tuple[str, float, float, float], ForwardStartPoint]:
    by_key: dict[tuple[str, float, float, float], ForwardStartPoint] = {}
    for point in points:
        if point.key in by_key:
            raise ValueError(
                f"La serie {name} contiene una fila repetida para "
                f"{point.option_type}, T1={point.t1:g}, T2={point.t2:g}, K={point.strike:g}."
            )
        by_key[point.key] = point
    return by_key


def compare_forward_start_sets(
    points_a: list[ForwardStartPoint],
    points_b: list[ForwardStartPoint],
    name_a: str = SERIE_A_NAME,
    name_b: str = SERIE_B_NAME,
    require_same_grid: bool = REQUIRE_SAME_GRID,
) -> list[ErrorRow]:
    """
    Alinea dos series por tipo, T1, T2 y strike, y calcula errores absolutos.
    """
    if not points_a:
        raise ValueError(f"La serie {name_a} no contiene precios.")
    if not points_b:
        raise ValueError(f"La serie {name_b} no contiene precios.")

    by_key_a = _points_by_key(points_a, name_a)
    by_key_b = _points_by_key(points_b, name_b)

    missing_in_b = sorted(set(by_key_a) - set(by_key_b))
    missing_in_a = sorted(set(by_key_b) - set(by_key_a))
    if require_same_grid and (missing_in_a or missing_in_b):
        messages = []
        if missing_in_b:
            messages.append(f"faltan en {name_b}: {_format_keys(missing_in_b)}")
        if missing_in_a:
            messages.append(f"faltan en {name_a}: {_format_keys(missing_in_a)}")
        raise ValueError("Las series no tienen los mismos puntos; " + "; ".join(messages))

    common_keys = sorted(set(by_key_a) & set(by_key_b), key=lambda item: (item[0], item[1], item[2], item[3]))
    if not common_keys:
        raise ValueError("Las series no tienen ningun punto comun para comparar.")

    rows: list[ErrorRow] = []
    for key in common_keys:
        point_a = by_key_a[key]
        point_b = by_key_b[key]
        rows.append(
            ErrorRow(
                option_type=point_a.option_type,
                t1=point_a.t1,
                t2=point_a.t2,
                strike=point_a.strike,
                price_a=point_a.price,
                price_b=point_b.price,
                abs_error=abs(point_a.price - point_b.price),
            )
        )
    return rows


def grid_differences(
    points_a: list[ForwardStartPoint],
    points_b: list[ForwardStartPoint],
    name_a: str = SERIE_A_NAME,
    name_b: str = SERIE_B_NAME,
) -> list[str]:
    by_key_a = _points_by_key(points_a, name_a)
    by_key_b = _points_by_key(points_b, name_b)
    messages = []
    missing_in_b = sorted(set(by_key_a) - set(by_key_b))
    missing_in_a = sorted(set(by_key_b) - set(by_key_a))
    if missing_in_b:
        messages.append(f"Puntos de {name_a} omitidos porque no estan en {name_b}: {_format_keys(missing_in_b)}")
    if missing_in_a:
        messages.append(f"Puntos de {name_b} omitidos porque no estan en {name_a}: {_format_keys(missing_in_a)}")
    return messages


def _format_keys(keys: list[tuple[str, float, float, float]], max_items: int = 8) -> str:
    shown = keys[:max_items]
    text = ", ".join(
        f"{tipo} T1={t1:g} T2={t2:g} K={strike:g}"
        for tipo, t1, t2, strike in shown
    )
    if len(keys) > max_items:
        text += f", ... ({len(keys)} en total)"
    return text


def summarize_by_pair(rows: list[ErrorRow]) -> list[dict[str, float | str | int]]:
    grouped: dict[tuple[str, float, float], list[ErrorRow]] = {}
    for row in rows:
        grouped.setdefault((row.option_type, round(row.t1, 10), round(row.t2, 10)), []).append(row)

    summary = []
    for (option_type, t1, t2), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        errors = [row.abs_error for row in group]
        summary.append(
            {
                "option_type": option_type,
                "t1": t1,
                "t2": t2,
                "n": len(errors),
                "mae": sum(errors) / len(errors),
                "max_error": max(errors),
            }
        )
    return summary


def summarize_global(rows: list[ErrorRow]) -> dict[str, float | int]:
    errors = [row.abs_error for row in rows]
    if not errors:
        raise ValueError("No hay errores para resumir.")
    return {
        "n": len(errors),
        "mae": sum(errors) / len(errors),
        "max_error": max(errors),
    }


def _format_float(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def print_error_rows(
    rows: list[ErrorRow],
    name_a: str,
    name_b: str,
    decimals: int,
) -> None:
    headers = ["Tipo", "T1", "T2", "K", f"Precio {name_a}", f"Precio {name_b}", "Error abs."]
    table = [
        [
            row.option_type,
            _format_float(row.t1, 2),
            _format_float(row.t2, 2),
            _format_float(row.strike, 2),
            _format_float(row.price_a, decimals),
            _format_float(row.price_b, decimals),
            _format_float(row.abs_error, decimals),
        ]
        for row in rows
    ]
    _print_table(headers, table)


def print_summary(summary: list[dict[str, float | str | int]], decimals: int) -> None:
    headers = ["Tipo", "T1", "T2", "N", "MAE", "Error maximo"]
    table = [
        [
            str(row["option_type"]),
            _format_float(float(row["t1"]), 2),
            _format_float(float(row["t2"]), 2),
            str(row["n"]),
            _format_float(float(row["mae"]), decimals),
            _format_float(float(row["max_error"]), decimals),
        ]
        for row in summary
    ]
    _print_table(headers, table)


def print_global_summary(summary: dict[str, float | int], decimals: int) -> None:
    headers = ["N", "MAE", "Error maximo"]
    table = [
        [
            str(summary["n"]),
            _format_float(float(summary["mae"]), decimals),
            _format_float(float(summary["max_error"]), decimals),
        ]
    ]
    _print_table(headers, table)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    print(" ".join(header.rjust(widths[index]) for index, header in enumerate(headers)))
    print(" ".join("-" * width for width in widths))
    for row in rows:
        print(" ".join(value.rjust(widths[index]) for index, value in enumerate(row)))


def _load_text(path: str | None, fallback: str) -> str:
    if path is None:
        return fallback
    return Path(path).read_text(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula MAE y error maximo entre dos series forward-start.",
    )
    parser.add_argument("--serie-a-file", help="Fichero con la salida de la primera serie.")
    parser.add_argument("--serie-b-file", help="Fichero con la salida de la segunda serie.")
    parser.add_argument("--serie-a-name", default=SERIE_A_NAME)
    parser.add_argument("--serie-b-name", default=SERIE_B_NAME)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument(
        "--estricto",
        action="store_true",
        default=REQUIRE_SAME_GRID,
        help="Exige que ambas series tengan exactamente los mismos T1, T2 y K.",
    )
    parser.add_argument(
        "--solo-resumen",
        action="store_true",
        help="Imprime solo metricas por par (T1,T2) y resumen global.",
    )
    return parser


def _points_from_config_or_files(
    args: argparse.Namespace,
) -> tuple[list[ForwardStartPoint], list[ForwardStartPoint]]:
    text_a = _load_text(args.serie_a_file, SERIE_A_TEXT)
    text_b = _load_text(args.serie_b_file, SERIE_B_TEXT)
    points_a = parse_forward_start_text(text_a)
    points_b = parse_forward_start_text(text_b)

    if points_a or points_b:
        return points_a, points_b

    if T1_T2_PAIRS and STRIKES and SERIE_A_PRICES and SERIE_B_PRICES:
        return (
            build_points_from_vectors(OPTION_TYPE, T1_T2_PAIRS, STRIKES, SERIE_A_PRICES),
            build_points_from_vectors(OPTION_TYPE, T1_T2_PAIRS, STRIKES, SERIE_B_PRICES),
        )

    raise ValueError(
        "No se han encontrado precios. Pega las tablas en SERIE_A_TEXT y SERIE_B_TEXT, "
        "rellena los vectores, o usa --serie-a-file y --serie-b-file."
    )


def main() -> None:
    args, _ = _build_parser().parse_known_args()
    points_a, points_b = _points_from_config_or_files(args)
    rows = compare_forward_start_sets(
        points_a,
        points_b,
        args.serie_a_name,
        args.serie_b_name,
        require_same_grid=args.estricto,
    )

    if not args.estricto:
        warnings = grid_differences(points_a, points_b, args.serie_a_name, args.serie_b_name)
        if warnings:
            print("=== Avisos de alineacion ===")
            for warning in warnings:
                print(warning)
            print()

    if not args.solo_resumen:
        print("=== Comparacion punto a punto forward-start ===")
        print_error_rows(rows, args.serie_a_name, args.serie_b_name, args.decimals)
        print()

    print("=== Metricas por par forward-start ===")
    print_summary(summarize_by_pair(rows), args.decimals)
    print()

    print("=== Metricas globales ===")
    print_global_summary(summarize_global(rows), args.decimals)


if __name__ == "__main__":
    main()
