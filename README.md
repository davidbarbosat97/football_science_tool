# Soccer Similarity Analytics Tool

Esta herramienta permite comparar el rendimiento de futbolistas profesionales pertenecientes a las 5 grandes ligas de Europa (La Liga, Premier League, Serie A, Bundesliga, Ligue 1) para encontrar los perfiles más similares utilizando algoritmos de Ciencia de Datos (distancia euclídea sobre métricas normalizadas por 90 minutos) y generar análisis tácticos avanzados por Inteligencia Artificial (OpenAI GPT).

> **Disponibilidad de datos:** la aplicación puede descargar datos de **Understat**, cargar CSV guardados dentro de `data/` o analizar un **CSV personalizado** subido desde la interfaz. Los datos de **FBref no están disponibles** porque su página web aplica restricciones que impiden realizar scraping de forma fiable y devuelve errores de acceso como `403 Forbidden`.

El proyecto está diseñado de forma modular, separando la lógica de extracción de datos, el motor matemático de recomendación y la interfaz de usuario. Esto permite escalar el sistema en el futuro, por ejemplo, cambiando de proveedor de datos, añadiendo otros algoritmos de similitud o reconstruyendo el frontend en tecnologías como React/TypeScript o Next.js.

---

## Características Principales

1. **Fuentes de Datos Escalables**: Permite descargar datos de **Understat** o subir bases de datos CSV extensas. La integración experimental con FBref permanece en el código, pero no está disponible debido a sus restricciones contra scraping.
2. **Normalización por 90 Minutos**: Todas las métricas acumuladas se dividen automáticamente por el tiempo de juego efectivo del futbolista, asegurando que las comparaciones sean equitativas y representen su rendimiento real por partido.
3. **Filtro de Minutos Dinámico**: Por defecto, descarta jugadores que no alcancen el 20% del máximo de minutos jugados en su liga para evitar desviaciones estadísticas debidas a muestras pequeñas. Es completamente personalizable en la interfaz.
4. **Filtros de Búsqueda**: Permite realizar búsquedas abiertas (comparar extremos con laterales, mediocentros con delanteros) o restringir las recomendaciones a posiciones específicas mediante un filtro interactivo en la interfaz.
5. **Gráfico de Radar Interactivo**: Utiliza Plotly para comparar mediante percentiles hasta 12 métricas seleccionables de todas las detectadas.
6. **Reporte Cualitativo por IA**: Llama a la API de OpenAI usando la clave configurada en `.env` para redactar un análisis de scouting detallado.

---

## Estructura del Proyecto

```
football_datascience_tool/
├── agents/
│   └── tactical_football_agent.py # Generación modular de reportes tácticos con OpenAI
├── data/                  # Directorio de caché local (guarda data en formato Parquet)
├── helpers/
│   ├── verify.py          # Verificación E2E con datos de Understat
│   ├── verify_csv.py      # Validación y diagnóstico de CSV externos
│   └── verify_fbref.py    # Diagnóstico experimental de acceso a FBref
├── src/
│   ├── data_loader.py     # Normalización y validación de datasets externos
│   ├── scraper.py         # Extracción Understat e integración experimental FBref
│   └── recommender.py     # Lógica matemática de similitud
├── app.py                 # Aplicación Streamlit (Frontend interactivo)
├── .env.example           # Plantilla de configuración para la API key
├── requirements.txt       # Listado de dependencias de Python
└── README.md              # Documentación e instrucciones
```

---

## Configuración e Instalación

### 1. Requisitos Previos
Asegúrate de tener instalado Python 3.9 o superior en tu sistema.

### 2. Clonar / Configurar Directorio
Establece este directorio como tu espacio de trabajo activo.

### 3. Crear Entorno Virtual e Instalar Dependencias
Ejecuta los siguientes comandos en tu terminal dentro de este directorio:

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Actualizar pip e instalar dependencias
pip install -U pip
pip install -r requirements.txt
```

### 4. Configurar OpenAI

Crea un archivo `.env` en la raíz del proyecto a partir de la plantilla:

```bash
cp .env.example .env
```

Después, sustituye el valor de `OPENAI_API_KEY` dentro de `.env` por tu clave.

---

## Uso de la Aplicación

Para iniciar el servidor de Streamlit y abrir la aplicación en tu navegador web, ejecuta:

```bash
streamlit run app.py
```

La aplicación se abrirá por defecto en `http://localhost:8501`.

### Pasos en la interfaz:
1. **Seleccionar Understat, CSV local o CSV personalizado** en el panel lateral. La primera descarga de Understat puede tardar unos segundos; después, los datos se cargarán desde la caché local.
2. **Configurar `OPENAI_API_KEY` en `.env`** para habilitar los reportes detallados por Inteligencia Artificial.
3. **Buscar un jugador** en la caja de búsqueda central.
4. **Ajustar filtros** de minutos jugados o posición de recomendación si lo deseas.
5. **Ver el Top 50** de jugadores más similares ordenados por porcentaje de afinidad matemática.
6. **Elegir un jugador** del listado para compararlo en el radar interactivo.
7. **Hacer clic en "Generar Reporte de Scouting IA"** para obtener el informe cualitativo completo de OpenAI.

---

## CSV Personalizados

La aplicación adapta automáticamente nombres de columnas habituales en español e inglés. Cada CSV debe incluir estos datos:

| Campo requerido | Algunos aliases reconocidos |
|---|---|
| Jugador | `player`, `Player`, `name`, `jugador` |
| Equipo | `team`, `Squad`, `club`, `equipo` |
| Liga | `league`, `Comp`, `competition`, `liga` |
| Posición | `position`, `Pos`, `posicion` |
| Minutos | `minutes`, `Min`, `mins`, `minutos` |
| Partidos | `matches`, `MP`, `apps`, `partidos` |

`age`, `born`, `nation` y `season` son opcionales. Si no existe `season`, se utiliza el valor indicado al subir el archivo.

Para seleccionar métricas estadísticas, el sistema:

- Descarta metadatos, rankings y columnas administrativas repetidas.
- Conserva columnas numéricas con cobertura global suficiente y variabilidad real.
- Para cada búsqueda, utiliza únicamente métricas con al menos un 50% de cobertura en la posición del jugador objetivo; así las estadísticas de porteros no distorsionan a los jugadores de campo.
- Respeta porcentajes y métricas que ya vienen calculadas por 90 minutos.
- Convierte automáticamente métricas acumuladas a valores por 90 minutos.
- Calcula únicamente la distancia del jugador objetivo contra el resto, evitando matrices cuadráticas en bases grandes.

Los CSV y Parquet colocados dentro de `data/` están excluidos de Git para evitar publicar bases privadas o pesadas.

El archivo `data/players_data-2024_2025_original.csv`, cuando está disponible localmente, aparece automáticamente como opción dentro de **CSV local**.

---

## Verificación

Para comprobar el flujo completo con Understat:

```bash
python helpers/verify.py
```

Para validar cualquier CSV y revisar las métricas que detectará el algoritmo:

```bash
python helpers/verify_csv.py /ruta/al/archivo.csv --season 2024-25
```

El siguiente diagnóstico permite comprobar las restricciones de acceso de FBref. No se considera un proveedor operativo:

```bash
python helpers/verify_fbref.py
```

---

## Escalabilidad del Proyecto

El código está desacoplado de la siguiente manera:
- **Añadir agentes de IA**: `TacticalFootballAgent` encapsula el prompt y la llamada al modelo. Nuevos agentes pueden añadirse bajo `agents/` y ser orquestados desde la aplicación o adaptados posteriormente a un framework.
- **Cambiar de proveedor de datos**: Puedes heredar de `BaseScraper` en `src/scraper.py` e implementar las funciones para un nuevo proveedor (como StatsBomb o Wyscout).
- **Modificar algoritmo de similitud**: El cálculo en `src/recommender.py` puede extenderse para usar otros métodos (como distancia de Mahalanobis, PCA + KNN, o ponderaciones manuales personalizadas de ciertas estadísticas dependiendo de la posición).
- **Desacoplar Frontend (JavaScript)**: Las clases `UnderstatScraper` y `PlayerRecommender` están listas para ser expuestas en una API de Python (usando FastAPI o Flask) a la cual un frontend en React, Vue o Next.js pueda consultar mediante llamadas HTTP, permitiendo una migración de interfaz limpia y moderna.
