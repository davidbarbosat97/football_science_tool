import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from io import BytesIO
from dotenv import load_dotenv
from agents.tactical_football_agent import TacticalFootballAgent
from src.data_loader import DatasetValidationError, PlayerDatasetNormalizer
from src.scraper import get_scraper
from src.recommender import PlayerRecommender

# Cargar variables de entorno del archivo .env
load_dotenv()

# Configuración de la página (Debe ser el primer comando de Streamlit)
st.set_page_config(
    page_title="Futbol Similarity Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilización premium con CSS personalizado inyectado
st.markdown("""
<style>
    /* Estilos globales */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fondo e inicio */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }

    /* Cards premium */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .metric-card-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #212529;
    }
    
    /* Reporte IA */
    .ia-report-box {
        background-color: #f0f4f8;
        border: 1px solid #d0e1fd;
        border-radius: 8px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Botón personalizado */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 6px;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Cabecera principal de la aplicación
st.markdown("""
<div class="main-header">
    <div class="main-title">⚽ Comparador Inteligente de Futbolistas</div>
    <div class="main-subtitle">Encuentra y analiza perfiles de jugadores similares utilizando Machine Learning e Inteligencia Artificial</div>
</div>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN DEL PANEL LATERAL (SIDEBAR) ---
st.sidebar.markdown("### ⚙️ Configuración y Filtros")

# La clave de OpenAI se obtiene exclusivamente desde el archivo .env.
openai_key = os.getenv("OPENAI_API_KEY", "")
if openai_key == "tu_api_key_de_openai_aqui":
    openai_key = ""
tactical_agent = TacticalFootballAgent(api_key=openai_key)

# 1. Selección de Fuente de Datos
local_csv_files = sorted(
    os.path.join("data", filename)
    for filename in os.listdir("data")
    if filename.lower().endswith(".csv")
) if os.path.isdir("data") else []
data_source_options = ["Understat"]
if local_csv_files:
    data_source_options.append("CSV local")
data_source_options.append("CSV personalizado")

data_source = st.sidebar.selectbox(
    "📊 Fuente de Datos",
    options=data_source_options,
    index=0,
    help="Understat descarga datos automáticamente. También puedes utilizar un CSV guardado en data/ o subir otro archivo."
)

all_leagues = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1"
]

local_csv_path = None
if data_source == "Understat":
    season = st.sidebar.selectbox(
        "📅 Temporada",
        options=["2025-26", "2024-25", "2023-24"],
        index=0
    )
    selected_leagues = st.sidebar.multiselect(
        "🏆 Ligas a Incluir",
        options=all_leagues,
        default=all_leagues
    )
    uploaded_csv = None
    csv_season = "CSV"
elif data_source == "CSV local":
    local_csv_path = st.sidebar.selectbox(
        "📁 Base de datos local",
        options=local_csv_files,
        format_func=os.path.basename,
    )
    uploaded_csv = None
    csv_season = st.sidebar.text_input("📅 Temporada del CSV", value="2024-25")
    selected_leagues = []
    season = csv_season
else:
    uploaded_csv = st.sidebar.file_uploader(
        "📁 Base de datos de jugadores",
        type=["csv"],
        help="Debe incluir jugador, equipo, liga, posición, minutos y partidos. Se reconocen aliases habituales en español e inglés."
    )
    csv_season = st.sidebar.text_input("📅 Temporada del CSV", value="CSV")
    selected_leagues = []
    season = csv_season

# 4. Slider de minutos mínimos (Porcentaje)
min_mins_pct = st.sidebar.slider(
    "⏱️ Minutos Mínimos Jugados (%)",
    min_value=5,
    max_value=80,
    value=20,
    step=5,
    help="Filtra a los jugadores que no alcancen este porcentaje de los minutos máximos jugados en su liga para evitar anomalías estadísticas."
)

force_reload = (
    st.sidebar.button("🔄 Recargar Datos de Internet")
    if data_source == "Understat"
    else False
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:gray;'>Proyecto de DataScience & Fútbol</small>", 
    unsafe_allow_html=True
)

# --- CARGA DE DATOS CON CACHÉ ---
@st.cache_data(show_spinner=False)
def load_data(prov: str, leagues: list, seas: str, reload: bool = False) -> pd.DataFrame:
    if not leagues:
        return pd.DataFrame()
    scraper = get_scraper(prov, leagues)
    # Si reload es True, forzamos recarga desde la web
    return scraper.load_season_data(seas, force_refresh=reload)

@st.cache_data(show_spinner=False)
def load_csv_data(file_bytes: bytes, default_season: str) -> pd.DataFrame:
    return PlayerDatasetNormalizer.read_csv(
        BytesIO(file_bytes), default_season=default_season
    )

@st.cache_data(show_spinner=False)
def load_local_csv_data(path: str, modified_at: float, default_season: str) -> pd.DataFrame:
    return PlayerDatasetNormalizer.read_csv(path, default_season=default_season)

# Estado de carga de datos
with st.spinner("Cargando y procesando la base de datos de futbolistas..."):
    try:
        if data_source == "Understat":
            df = load_data("Understat", selected_leagues, season, reload=force_reload)
        elif data_source == "CSV local" and local_csv_path:
            df = load_local_csv_data(
                local_csv_path, os.path.getmtime(local_csv_path), csv_season
            )
        elif uploaded_csv is not None:
            df = load_csv_data(uploaded_csv.getvalue(), csv_season)
        else:
            df = pd.DataFrame()

        if data_source in {"CSV local", "CSV personalizado"} and not df.empty:
            csv_seasons = sorted(df["season"].dropna().unique().tolist(), reverse=True)
            selected_csv_season = st.sidebar.selectbox(
                "📅 Temporada disponible",
                options=csv_seasons,
            )
            df = df[df["season"] == selected_csv_season].copy()
            csv_leagues = sorted(df["league"].dropna().unique().tolist())
            selected_csv_leagues = st.sidebar.multiselect(
                "🏆 Ligas del CSV",
                options=csv_leagues,
                default=csv_leagues,
            )
            df = df[df["league"].isin(selected_csv_leagues)].copy()
    except DatasetValidationError as e:
        st.error(f"El CSV no tiene un formato compatible: {e}")
        df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar los datos desde {data_source}: {e}")
        df = pd.DataFrame()

# --- FLUJO PRINCIPAL DE LA APLICACIÓN ---
if df.empty:
    if data_source == "CSV personalizado" and uploaded_csv is None:
        st.info("Sube un archivo CSV para comenzar el análisis.")
    else:
        st.warning("⚠️ La base de datos está vacía. Comprueba la fuente y los filtros seleccionados.")
else:
    # Inicializar el recomendador
    recommender = PlayerRecommender(df)
    st.sidebar.caption(
        f"{len(recommender.df_processed):,} jugadores · "
        f"{len(recommender.feature_cols):,} métricas válidas"
    )
    with st.sidebar.expander("Métricas detectadas"):
        st.write(", ".join(recommender.feature_cols))
    
    # Crear opciones formateadas para buscar jugadores
    # "Nombre (Equipo, Liga)" para evitar confusiones de homónimos
    df_search = recommender.df_processed.copy()
    df_search["search_display"] = df_search["player"] + " (" + df_search["team"] + ", " + df_search["league"] + ")"
    search_options = sorted(df_search["search_display"].tolist())
    
    st.markdown("### 🔍 Busca un Futbolista")
    selected_search = st.selectbox(
        "Escribe el nombre del jugador que quieres comparar:",
        options=search_options,
        index=0
    )
    
    # Extraer el jugador seleccionado
    target_row = df_search[df_search["search_display"] == selected_search].iloc[0]
    target_name = target_row["player"]
    target_team = target_row["team"]
    
    # Mostrar tarjeta del jugador seleccionado
    st.markdown("#### Jugador Seleccionado")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Jugador</div>
            <div class="metric-card-value">{target_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Equipo / Liga</div>
            <div class="metric-card-value" style="font-size: 1.1rem; padding-top: 0.3rem;">{target_team}<br><small style="color:gray;">{target_row['league']}</small></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Posición / Edad</div>
            <div class="metric-card-value">{target_row['clean_position']} <span style="font-size:1rem; font-weight:normal;">({target_row.get('age', 'N/A')} años)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Minutos Jugados</div>
            <div class="metric-card-value">{int(target_row['minutes'])} mins</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- RESULTADOS DE SIMILITUD ---
    st.markdown("### 🏆 Futbolistas Más Similares (Top 50)")
    
    # Selector de filtro por posición interactivamente
    pos_filter = st.selectbox(
        "Filtrar lista de recomendación por posición:",
        options=["Todos", "Delantero", "Centrocampista", "Defensa", "Portero"],
        index=0
    )
    
    try:
        similar_df = recommender.find_similar_players(
            player_name=target_name,
            team_name=target_team,
            league_name=target_row["league"],
            season=target_row["season"],
            min_minutes_pct=min_mins_pct,
            position_filter=pos_filter,
            top_n=50
        )
    except Exception as e:
        st.error(f"Error al calcular similitudes: {e}")
        similar_df = pd.DataFrame()
        
    if similar_df.empty:
        st.info("No se encontraron jugadores similares con los filtros actuales. Prueba a disminuir el umbral de minutos jugados o quitar los filtros de posición.")
    else:
        # Formatear el DataFrame para visualización amigable
        display_df = similar_df.copy()
        display_df["Similitud (%)"] = (display_df["similarity"] * 100).round(1)
        display_df["Edad"] = display_df.get(
            "age", pd.Series("N/A", index=display_df.index)
        ).fillna("N/A")
        display_df["Minutos"] = display_df["minutes"].astype(int)
        display_df["Partidos"] = display_df["matches"].astype(int)
        
        # Resetear índice para mostrar numeración del 1 al 50
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        
        # Mostrar tabla interactiva
        st.dataframe(
            display_df[["player", "team", "league", "clean_position", "Edad", "Minutos", "Partidos", "Similitud (%)"]].rename(columns={
                "player": "Jugador",
                "team": "Equipo",
                "league": "Liga",
                "clean_position": "Posición"
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # --- COMPARACIÓN DETALLADA & GRÁFICO DE RADAR ---
        st.markdown("### 📊 Comparación Detallada")
        
        # Permitir al usuario elegir con cuál del top 50 comparar en detalle
        compare_options = display_df["player"] + " (" + display_df["team"] + ")"
        selected_compare = st.selectbox(
            "Selecciona un jugador similar de la lista para comparar en detalle:",
            options=compare_options
        )
        
        # Encontrar fila del jugador a comparar
        compare_player_name = selected_compare.split(" (")[0]
        compare_player_team = selected_compare.split(" (")[1].replace(")", "")
        compare_row = display_df[
            (display_df["player"] == compare_player_name) & 
            (display_df["team"] == compare_player_team)
        ].iloc[0]
        
        # Columnas de comparación visual
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Comparación Visual (Radar Chart)")
            
            # Mostrar las mismas métricas utilizadas por el algoritmo de similitud.
            radar_label_map = {
                "goals_per90": "Goles",
                "xg_per90": "xG",
                "np_goals_per90": "Goles sin penalti",
                "np_xg_per90": "npxG",
                "assists_per90": "Asistencias",
                "xa_per90": "xA",
                "shots_per90": "Tiros",
                "key_passes_per90": "Pases clave",
                "yellow_cards_per90": "Tarjetas amarillas",
                "red_cards_per90": "Tarjetas rojas",
                "xg_chain_per90": "xG Chain",
                "xg_buildup_per90": "xG Buildup",
            }
            target_feature_cols = recommender.get_features_for_position(
                target_row["clean_position"]
            )
            default_radar_features = target_feature_cols[:12]
            if len(target_feature_cols) > 12:
                selected_radar_features = st.multiselect(
                    "Métricas visibles en el radar (máximo 12):",
                    options=target_feature_cols,
                    default=default_radar_features,
                    format_func=lambda feat: radar_label_map.get(
                        feat, recommender.feature_label(feat)
                    ),
                    max_selections=12,
                )
            else:
                selected_radar_features = default_radar_features
            radar_features = [
                (radar_label_map.get(feat, recommender.feature_label(feat)), feat)
                for feat in selected_radar_features
            ]
            
            # Filtrar métricas que realmente existan en el dataset
            valid_radar_labels = []
            valid_radar_features = []
            radar_df = recommender.df_processed
            radar_reference_df = recommender.get_filtered_dataset(min_mins_pct)
            radar_reference_df = radar_reference_df[
                radar_reference_df["clean_position"] == target_row["clean_position"]
            ]
            for label, feat in radar_features:
                if feat in radar_df.columns:
                    valid_radar_labels.append(label)
                    valid_radar_features.append(feat)
            
            if not valid_radar_features:
                st.warning("No hay suficientes métricas coincidentes en este proveedor de datos para generar el radar.")
            else:
                # Obtener valores
                values_target = []
                values_compare = []
                
                # Representar cada métrica como percentil frente a jugadores con
                # suficientes minutos evita que valores extremos aplasten el radar.
                for feat in valid_radar_features:
                    val_t = target_row[feat]
                    val_c = compare_row[feat]
                    reference_values = radar_reference_df[feat].dropna()
                    norm_t = (reference_values <= val_t).mean()
                    norm_c = (reference_values <= val_c).mean()
                    
                    # Añadimos además el valor real para mostrar en el tooltip
                    values_target.append((norm_t, val_t))
                    values_compare.append((norm_c, val_c))
                
                # Crear gráfico de radar
                fig = go.Figure()
                closed_labels = valid_radar_labels + [valid_radar_labels[0]]
                closed_target = values_target + [values_target[0]]
                closed_compare = values_compare + [values_compare[0]]
                
                # Target
                fig.add_trace(go.Scatterpolar(
                    r=[v[0] for v in closed_target],
                    theta=closed_labels,
                    fill='toself',
                    name=target_name,
                    line=dict(color='#2563eb', width=3),
                    fillcolor='rgba(37, 99, 235, 0.22)',
                    hoverinfo="text",
                    text=[f"{label}: {v[1]:.2f}" for label, v in zip(closed_labels, closed_target)]
                ))
                
                # Compare
                fig.add_trace(go.Scatterpolar(
                    r=[v[0] for v in closed_compare],
                    theta=closed_labels,
                    fill='toself',
                    name=compare_player_name,
                    line=dict(color='#f97316', width=3),
                    fillcolor='rgba(249, 115, 22, 0.22)',
                    hoverinfo="text",
                    text=[f"{label}: {v[1]:.2f}" for label, v in zip(closed_labels, closed_compare)]
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickvals=[0.2, 0.4, 0.6, 0.8, 1],
                            ticktext=["P20", "P40", "P60", "P80", "P100"],
                            showticklabels=True
                        )
                    ),
                    showlegend=True,
                    margin=dict(l=85, r=0, t=20, b=20),
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("#### 🤖 Análisis Táctico con Inteligencia Artificial")
            st.markdown("Presiona el botón para generar un informe detallado comparando los estilos y encajes tácticos de ambos jugadores.")
            
            # Botón para solicitar el reporte al agente táctico
            generate_report = st.button("🧠 Generar Reporte de Scouting IA")
            
            if generate_report:
                if not openai_key:
                    st.warning("⚠️ Se requiere una **OpenAI API Key** para esta función. Configura `OPENAI_API_KEY` en el archivo `.env`.")
                else:
                    with st.spinner(f"Analizando perfiles de {target_name} y {compare_player_name}..."):
                        report_text = tactical_agent.generate_report(
                            player_a=target_row,
                            player_b=compare_row,
                            feature_cols=recommender.feature_cols,
                        )
                        st.markdown(f"""
                        <div class="ia-report-box">
                            {report_text.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
