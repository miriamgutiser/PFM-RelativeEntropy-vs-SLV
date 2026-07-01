"""
Este módulo proporciona funciones para exportar variables a un archivo .py y luego importarlas como un módulo de 
Python. 

"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def _numpy_dtype_literal(dtype: np.dtype[Any]) -> str:
    numpy_dtype = np.dtype(dtype)
    dtype_name = numpy_dtype.name
    if hasattr(np, dtype_name):
        return f"np.{dtype_name}"
    return f"np.dtype({numpy_dtype.str!r})"


def _python_literal(value: Any) -> str:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return (
            f"np.array({repr(array.tolist())}, dtype={_numpy_dtype_literal(array.dtype)})"
        )

    if isinstance(value, np.generic):
        return repr(value.item())

    if isinstance(value, (float, int, bool, str)):
        return repr(value)

    array = np.asarray(value)
    if array.ndim == 0:
        return repr(array.item())
    return f"np.array({repr(array.tolist())}, dtype={_numpy_dtype_literal(array.dtype)})"


def export_python_variables_module(
    module_path: Path,
    variables: dict[str, Any],
) -> Path:
    """
    Guarda variables en un archivo .py para poder importarlas despues.
    """
    module_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = module_path.with_name(f"{module_path.name}.tmp")
    lines = [
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
    ]

    for name, value in variables.items():
        lines.append(f"{name} = {_python_literal(value)}")
        lines.append("")

    temp_path.write_text("\n".join(lines), encoding="utf-8")
    temp_path.replace(module_path)
    return module_path


def load_python_variables_module(module_path: Path) -> ModuleType:
    """
    Carga un archivo .py generado con variables y devuelve el modulo.
    """
    if not module_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de variables: {module_path}")

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear el loader para {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
