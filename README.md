# PFM-RelativeEntropy vs SLV

Este repositorio contiene el trabajo de mi proyecto fin de master en Finanzas, realizado en colaboración con el banco Santander.

El objetivo del trabajo es el estudio comparativo entre enfoques basados en Entropía Relativa
y modelos de Volatilidad Estocástica–Local aplicados a la modelización y valoración de derivados financieros.

## Estructura del repositorio

- `docs/`: documentación del proyecto y artículos científicos de referencia.
- `src/`: código fuente 
- `data/`: conjuntos de datos utilizados en el trabajo
- `results/`: resultados numéricos, gráficos y tablas
- `meetings/`: actas y notas de reuniones de seguimiento

## Instalación

El código fuente está organizado como un paquete Python instalable. Para ejecutar cualquier script del proyecto es necesario instalarlo previamente en el entorno de Python.

### Requisitos previos

- Python 3.9 o superior
- `pip` (incluido con Python)

Las dependencias del proyecto (`numpy`, `scipy`, `matplotlib`,...) se instalan automáticamente.

### Pasos

**1. Clonar el repositorio**

```bash
git clone <url-del-repositorio>
cd PFM-RelativeEntropy-vs-SLV
```

**2. (Opcional) Crear un entorno virtual**

Se recomienda para aislar las dependencias del proyecto:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

**3. Instalar el paquete en modo editable**

```bash
pip install -e .
```

El flag `-e` (*editable*) enlaza directamente con el código fuente, por lo que cualquier modificación en `src/` se refleja inmediatamente sin necesidad de reinstalar.

**4. Verificar la instalación**

```bash
python -c "from core.types import OptionType; print('OK')"
```

Si aparece `OK`, el entorno está configurado correctamente y todos los scripts de `src/` pueden ejecutarse directamente.

## Estado
Trabajo en curso
