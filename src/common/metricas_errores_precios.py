"""
Calcula MAE y error maximo entre dos series de precios de opciones.

Uso rapido:
1. Pega la salida de slv_precios_final.py en SERIE_A_TEXT.
2. Pega la salida de wmc_precios.py en SERIE_B_TEXT.
3. Ejecuta:

   python src/common/metricas_errores_precios.py

Tambien se pueden pasar ficheros de texto con --serie-a-file y --serie-b-file.
El parser usa solo las filas que empiezan por CALL o PUT, por ejemplo:

CALL      70.00         0.08          30.00
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

SERIE_A_TEXT = """
Tipo     Strike  Vencimiento     Precio WMC
---------------------------------------------
CALL      70.00         0.02          30.00
CALL      90.00         0.02          10.00
CALL     110.00         0.02           0.00
CALL      70.00         0.08          30.00
CALL      90.00         0.08          10.00
CALL     110.00         0.08           0.00
CALL      70.00         0.50          30.00
CALL      90.00         0.50          10.01
CALL     110.00         0.50           0.07
CALL      70.00         1.00          30.00
CALL      90.00         1.00          10.12
CALL     110.00         1.00           0.36
CALL      70.00         2.00          30.00
CALL      90.00         2.00          10.53
CALL     110.00         2.00           0.70
CALL      70.00         5.00          30.05
CALL      90.00         5.00          12.16
CALL     110.00         5.00           2.30
CALL      70.00         8.00          30.28
CALL      90.00         8.00          13.72
CALL     110.00         8.00           4.10
CALL      70.00        10.00          30.55
CALL      90.00        10.00          14.70
CALL     110.00        10.00           5.25
"""

SERIE_B_TEXT = """
=== Precios Heston-SLV finales ===
Tipo     Strike  Vencimiento     Precio SLV
---------------------------------------------
CALL      70.00         0.02          30.00
CALL      90.00         0.02          10.00
CALL     110.00         0.02           0.00
CALL      70.00         0.08          30.00
CALL      90.00         0.08          10.00
CALL     110.00         0.08           0.00
CALL      70.00         0.50          30.04
CALL      90.00         0.50          10.10
CALL     110.00         0.50           0.04
CALL      70.00         1.00          30.06
CALL      90.00         1.00          10.24
CALL     110.00         1.00           0.07
CALL      70.00         2.00          30.09
CALL      90.00         2.00          10.81
CALL     110.00         2.00           0.50
CALL      70.00         5.00          30.47
CALL      90.00         5.00          12.58
CALL     110.00         5.00           2.26
CALL      70.00         8.00          30.94
CALL      90.00         8.00          14.07
CALL     110.00         8.00           4.06
CALL      70.00        10.00          31.22
CALL      90.00        10.00          14.97
CALL     110.00        10.00           5.20
"""


# Alternativa: rellena vectores en orden vencimiento -> strike.
# Ejemplo con 8 vencimientos y 3 strikes: 24 precios por serie.
OPTION_TYPE = "CALL"
STRIKES: tuple[float, ...] = ()
MATURITIES: tuple[float, ...] = ()
SERIE_A_PRICES: tuple[float, ...] | tuple[tuple[float, ...], ...] = ()
SERIE_B_PRICES: tuple[float, ...] | tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class PricePoint:
    option_type: str
    strike: float
    maturity: float
    price: float

    @property
    def key(self) -> tuple[str, float, float]:
        return (self.option_type.upper(), round(self.strike, 10), round(self.maturity, 10))


@dataclass(frozen=True)
class ErrorRow:
    option_type: str
    strike: float
    maturity: float
    price_a: float
    price_b: float
    abs_error: float


def _parse_float(value: str) -> float:
    return float(value.strip().replace(",", "."))


def parse_price_text(text: str) -> list[PricePoint]:
    """
    Extrae filas tipo ``CALL strike maturity price`` o ``PUT strike maturity price``.
    """
    points: list[PricePoint] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().replace("|", " ")
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        option_type = parts[0].upper()
        if option_type not in {"CALL", "PUT"}:
            continue

        try:
            strike = _parse_float(parts[1])
            maturity = _parse_float(parts[2])
            price = _parse_float(parts[3])
        except ValueError:
            continue

        points.append(
            PricePoint(
                option_type=option_type,
                strike=strike,
                maturity=maturity,
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
    strikes: Iterable[float],
    maturities: Iterable[float],
    prices: Iterable,
) -> list[PricePoint]:
    """
    Construye puntos desde precios en orden: para cada vencimiento, todos los strikes.
    """
    strike_values = [float(strike) for strike in strikes]
    maturity_values = [float(maturity) for maturity in maturities]
    price_values = _flatten_price_values(prices)
    expected = len(strike_values) * len(maturity_values)
    if len(price_values) != expected:
        raise ValueError(
            "La serie de precios debe tener len(maturities) * len(strikes) valores. "
            f"Esperado: {expected}; recibido: {len(price_values)}."
        )

    normalized_type = option_type.strip().upper()
    if normalized_type not in {"CALL", "PUT"}:
        raise ValueError("option_type debe ser CALL o PUT.")

    points: list[PricePoint] = []
    index = 0
    for maturity in maturity_values:
        for strike in strike_values:
            points.append(
                PricePoint(
                    option_type=normalized_type,
                    strike=strike,
                    maturity=maturity,
                    price=float(price_values[index]),
                )
            )
            index += 1
    return points


def _points_by_key(points: list[PricePoint], name: str) -> dict[tuple[str, float, float], PricePoint]:
    by_key: dict[tuple[str, float, float], PricePoint] = {}
    for point in points:
        if point.key in by_key:
            raise ValueError(
                f"La serie {name} contiene una fila repetida para "
                f"{point.option_type}, K={point.strike:g}, T={point.maturity:g}."
            )
        by_key[point.key] = point
    return by_key


def compare_price_sets(
    points_a: list[PricePoint],
    points_b: list[PricePoint],
    name_a: str = SERIE_A_NAME,
    name_b: str = SERIE_B_NAME,
) -> list[ErrorRow]:
    """
    Alinea dos series por tipo, strike y vencimiento, y calcula errores absolutos.
    """
    if not points_a:
        raise ValueError(f"La serie {name_a} no contiene precios.")
    if not points_b:
        raise ValueError(f"La serie {name_b} no contiene precios.")

    by_key_a = _points_by_key(points_a, name_a)
    by_key_b = _points_by_key(points_b, name_b)

    missing_in_b = sorted(set(by_key_a) - set(by_key_b))
    missing_in_a = sorted(set(by_key_b) - set(by_key_a))
    if missing_in_a or missing_in_b:
        messages = []
        if missing_in_b:
            messages.append(f"faltan en {name_b}: {_format_keys(missing_in_b)}")
        if missing_in_a:
            messages.append(f"faltan en {name_a}: {_format_keys(missing_in_a)}")
        raise ValueError("Las series no tienen los mismos puntos; " + "; ".join(messages))

    rows: list[ErrorRow] = []
    for key in sorted(by_key_a, key=lambda item: (item[0], item[2], item[1])):
        point_a = by_key_a[key]
        point_b = by_key_b[key]
        rows.append(
            ErrorRow(
                option_type=point_a.option_type,
                strike=point_a.strike,
                maturity=point_a.maturity,
                price_a=point_a.price,
                price_b=point_b.price,
                abs_error=abs(point_a.price - point_b.price),
            )
        )
    return rows


def _format_keys(keys: list[tuple[str, float, float]], max_items: int = 8) -> str:
    shown = keys[:max_items]
    text = ", ".join(f"{tipo} K={strike:g} T={maturity:g}" for tipo, strike, maturity in shown)
    if len(keys) > max_items:
        text += f", ... ({len(keys)} en total)"
    return text


def summarize_by_maturity(rows: list[ErrorRow]) -> list[dict[str, float | str | int]]:
    grouped: dict[tuple[str, float], list[ErrorRow]] = {}
    for row in rows:
        grouped.setdefault((row.option_type, round(row.maturity, 10)), []).append(row)

    summary = []
    for (option_type, maturity), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        errors = [row.abs_error for row in group]
        summary.append(
            {
                "option_type": option_type,
                "maturity": maturity,
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
    headers = ["Tipo", "Strike", "Vencimiento", f"Precio {name_a}", f"Precio {name_b}", "Error abs."]
    table = [
        [
            row.option_type,
            _format_float(row.strike, 2),
            _format_float(row.maturity, 2),
            _format_float(row.price_a, decimals),
            _format_float(row.price_b, decimals),
            _format_float(row.abs_error, decimals),
        ]
        for row in rows
    ]
    _print_table(headers, table)


def print_summary(summary: list[dict[str, float | str | int]], decimals: int) -> None:
    headers = ["Tipo", "Vencimiento", "N", "MAE", "Error maximo"]
    table = [
        [
            str(row["option_type"]),
            _format_float(float(row["maturity"]), 2),
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
        description="Calcula MAE y error maximo entre dos series de precios.",
    )
    parser.add_argument("--serie-a-file", help="Fichero con la salida de la primera serie.")
    parser.add_argument("--serie-b-file", help="Fichero con la salida de la segunda serie.")
    parser.add_argument("--serie-a-name", default=SERIE_A_NAME)
    parser.add_argument("--serie-b-name", default=SERIE_B_NAME)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument(
        "--solo-resumen",
        action="store_true",
        help="Imprime solo metricas por vencimiento y resumen global.",
    )
    return parser


def _points_from_config_or_files(args: argparse.Namespace) -> tuple[list[PricePoint], list[PricePoint]]:
    text_a = _load_text(args.serie_a_file, SERIE_A_TEXT)
    text_b = _load_text(args.serie_b_file, SERIE_B_TEXT)
    points_a = parse_price_text(text_a)
    points_b = parse_price_text(text_b)

    if points_a or points_b:
        return points_a, points_b

    if STRIKES and MATURITIES and SERIE_A_PRICES and SERIE_B_PRICES:
        return (
            build_points_from_vectors(OPTION_TYPE, STRIKES, MATURITIES, SERIE_A_PRICES),
            build_points_from_vectors(OPTION_TYPE, STRIKES, MATURITIES, SERIE_B_PRICES),
        )

    raise ValueError(
        "No se han encontrado precios. Pega las tablas en SERIE_A_TEXT y SERIE_B_TEXT, "
        "rellena los vectores, o usa --serie-a-file y --serie-b-file."
    )


def main() -> None:
    args, _ = _build_parser().parse_known_args()
    points_a, points_b = _points_from_config_or_files(args)
    rows = compare_price_sets(points_a, points_b, args.serie_a_name, args.serie_b_name)

    if not args.solo_resumen:
        print("=== Comparacion punto a punto ===")
        print_error_rows(rows, args.serie_a_name, args.serie_b_name, args.decimals)
        print()

    print("=== Metricas por vencimiento ===")
    print_summary(summarize_by_maturity(rows), args.decimals)
    print()

    print("=== Metricas globales ===")
    print_global_summary(summarize_global(rows), args.decimals)


if __name__ == "__main__":
    main()
