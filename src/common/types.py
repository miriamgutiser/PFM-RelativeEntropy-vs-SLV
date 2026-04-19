"""
===============================================================================
OPTION TYPE ENUMERATION
===============================================================================
Este módulo define el tipo de opción financiera utilizada en los modelos
de valoración.

Incluye:
    - OptionType (CALL / PUT)

Notas:
    - Se implementa como enumeración (enum.Enum).
    - CALL y PUT se representan numéricamente como +1 y -1 respectivamente.
"""

import enum

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
