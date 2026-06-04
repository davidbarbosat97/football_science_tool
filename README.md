# Soccer Similarity Analytics Tool

Esta herramienta permite comparar el rendimiento de futbolistas profesionales pertenecientes a las 5 grandes ligas de Europa (La Liga, Premier League, Serie A, Bundesliga, Ligue 1) para encontrar los perfiles más similares utilizando algoritmos de Ciencia de Datos (distancia euclídea sobre métricas normalizadas por 90 minutos) y generar análisis tácticos avanzados por Inteligencia Artificial (OpenAI GPT).

El proyecto está diseñado de forma modular, separando la lógica de extracción de datos, el motor matemático de recomendación y la interfaz de usuario. Esto permite escalar el sistema en el futuro, por ejemplo, cambiando de proveedor de datos, añadiendo otros algoritmos de similitud o reconstruyendo el frontend en tecnologías como React/TypeScript o Next.js.

---

## Características Principales

1. **Extracción y Caché Local**: Integración con la librería `soccerdata`. Soporta:
   - **Understat**: (Recomendado) Muy veloz, excelente estabilidad sin bloqueos de IP, e incluye métricas avanzadas esperadas (xG, xA, xG Chain, xG Buildup).
   - **FBref**: Mayor amplitud de estadísticas (tiros, pases, regates, entradas, intercepciones, despejes), ideal si se ejecuta desde redes residenciales (puede dar error de bloqueo 403 Forbidden en servidores o redes datacenters).
2. **Normalización por 90 Minutos**: Todas las métricas acumuladas se dividen automáticamente por el tiempo de juego efectivo del futbolista, asegurando que las comparaciones sean equitativas y representen su rendimiento real por partido.
3. **Filtro de Minutos Dinámico**: Por defecto, descarta jugadores que no alcancen el 20% del máximo de minutos jugados en su liga para evitar desviaciones estadísticas debidas a muestras pequeñas. Es completamente personalizable en la interfaz.
4. **Filtros de Búsqueda**: Permite realizar búsquedas abiertas (comparar extremos con laterales, mediocentros con delanteros) o restringir las recomendaciones a posiciones específicas mediante un filtro interactivo en la interfaz.
5. **Gráfico de Radar Interactivo**: Utiliza Plotly para comparar visualmente las 12 métricas empleadas por el algoritmo mediante percentiles.
6. **Reporte Cualitativo por IA**: Llama a la API de OpenAI usando la clave configurada en `.env` para redactar un análisis de scouting detallado.

---

## Estructura del Proyecto

```
football_datascience_tool/
├── data/                  # Directorio de caché local (guarda data en formato Parquet)
├── helpers/
│   ├── verify.py          # Verificación E2E con datos de Understat
│   └── verify_fbref.py    # Verificación de conexión y descarga desde FBref
├── src/
│   ├── scraper.py         # Módulo extractor de datos (FBref y Understat)
│   └── recommender.py     # Lógica matemática de similitud y conexión con OpenAI
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
1. **Seleccionar Proveedor y Temporada** en el panel lateral (la primera vez que selecciones una temporada/proveedor, el sistema descargará los datos y tardará unos segundos. A partir de ahí, se cargarán instantáneamente desde la caché local).
2. **Configurar `OPENAI_API_KEY` en `.env`** para habilitar los reportes detallados por Inteligencia Artificial.
3. **Buscar un jugador** en la caja de búsqueda central.
4. **Ajustar filtros** de minutos jugados o posición de recomendación si lo deseas.
5. **Ver el Top 50** de jugadores más similares ordenados por porcentaje de afinidad matemática.
6. **Elegir un jugador** del listado para compararlo en el radar interactivo.
7. **Hacer clic en "Generar Reporte de Scouting IA"** para obtener el informe cualitativo completo de OpenAI.

---

## Verificación

Para comprobar el flujo completo con Understat:

```bash
python helpers/verify.py
```

Para probar específicamente la conexión y descarga desde FBref:

```bash
python helpers/verify_fbref.py
```

---

## Escalabilidad del Proyecto

El código está desacoplado de la siguiente manera:
- **Cambiar de proveedor de datos**: Puedes heredar de `BaseScraper` en `src/scraper.py` e implementar las funciones para un nuevo proveedor (como StatsBomb o Wyscout).
- **Modificar algoritmo de similitud**: El cálculo en `src/recommender.py` puede extenderse para usar otros métodos (como distancia de Mahalanobis, PCA + KNN, o ponderaciones manuales personalizadas de ciertas estadísticas dependiendo de la posición).
- **Desacoplar Frontend (JavaScript)**: Las clases `FBrefScraper`, `UnderstatScraper` y `PlayerRecommender` están listas para ser expuestas en una API de Python (usando FastAPI o Flask) a la cual un frontend en React, Vue o Next.js pueda consultar mediante llamadas HTTP, permitiendo una migración de interfaz limpia y moderna.
