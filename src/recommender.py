import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from openai import OpenAI
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class PlayerRecommender:
    """
    Motor de similitud y recomendación de futbolistas.
    Preprocesa los datos, calcula métricas por 90 minutos, aplica distancia euclídea
    e interactúa con OpenAI para el análisis cualitativo.
    """
    
    METADATA_COLS = [
        "league", "season", "team", "player", "position", "age", "born", "nation",
        "minutes", "matches", "league_id", "season_id", "team_id", "player_id"
    ]
    OPTIONAL_METADATA_DEFAULTS = {
        "age": "N/A",
        "born": "N/A",
        "nation": "N/A",
    }

    def __init__(self, df: pd.DataFrame):
        self.df_raw = df.copy()
        self.df_processed = pd.DataFrame()
        self.feature_cols = []
        self.scaler = MinMaxScaler()
        self._prepare_data()

    def _standardize_position(self, pos_str: str) -> str:
        """
        Estandariza los códigos de posición de FBref y Understat a categorías generales en español.
        """
        if not isinstance(pos_str, str):
            return "Desconocido"
        
        pos_str = pos_str.upper()
        # Mapeo general
        if "GK" in pos_str or "GOALKEEPER" in pos_str:
            return "Portero"
        elif "DF" in pos_str or "DEFENDER" in pos_str or "D" == pos_str or pos_str.startswith("D "):
            return "Defensa"
        elif "MF" in pos_str or "MIDFIELDER" in pos_str or "M" == pos_str or pos_str.startswith("M "):
            return "Centrocampista"
        elif "FW" in pos_str or "FORWARD" in pos_str or "F" == pos_str or pos_str.startswith("F "):
            return "Delantero"
        
        # Fallbacks específicos para posiciones combinadas en Understat (ej. "M S", "F S")
        if pos_str.startswith("D"):
            return "Defensa"
        elif pos_str.startswith("M"):
            return "Centrocampista"
        elif pos_str.startswith("F"):
            return "Delantero"
            
        return "Centrocampista" # Fallback por defecto

    def _prepare_data(self):
        """
        Limpia los datos, estandariza posiciones, calcula métricas por 90 minutos
        y prepara la matriz de características.
        """
        if self.df_raw.empty:
            return

        df = self.df_raw.copy()

        # Algunos proveedores, como Understat, no incluyen todos los metadatos.
        for column, default_value in self.OPTIONAL_METADATA_DEFAULTS.items():
            if column not in df.columns:
                df[column] = default_value
        
        # 1. Asegurar tipos de datos básicos
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
        df["matches"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)
        
        # 2. Estandarizar la columna de posición
        df["clean_position"] = df["position"].apply(self._standardize_position)
        
        # 3. Filtrar registros con minutos inválidos
        df = df[df["minutes"] > 0].copy()

        # 4. Seleccionar columnas numéricas que representan estadísticas
        all_cols = df.columns
        exclude = set(self.METADATA_COLS + ["clean_position"])
        numeric_stat_cols = [
            col for col in all_cols 
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # 5. Convertir estadísticas brutas a métricas por 90 minutos
        # Evita normalizar columnas de porcentaje (ej. que contengan "%" o "pct" o "rate")
        df_p90 = df.copy()
        self.feature_cols = []
        
        for col in numeric_stat_cols:
            col_lower = col.lower()
            is_percentage = "%" in col or "pct" in col or "rate" in col or "percent" in col
            is_average = "avg" in col_lower or "average" in col_lower or "per" in col_lower
            
            if is_percentage or is_average:
                # Mantener porcentajes y promedios directamente
                p90_col_name = col
                df_p90[p90_col_name] = df[col].fillna(0)
            else:
                # Calcular por 90 minutos
                p90_col_name = f"{col}_per90"
                df_p90[p90_col_name] = (df[col] / (df["minutes"] / 90.0)).fillna(0)
                
            self.feature_cols.append(p90_col_name)

        # Reemplazar valores infinitos (por división por cero) por 0
        df_p90[self.feature_cols] = df_p90[self.feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        
        self.df_processed = df_p90

    def get_filtered_dataset(self, min_minutes_pct: float) -> pd.DataFrame:
        """
        Retorna el dataset procesado filtrando por el porcentaje mínimo de minutos jugados.
        El porcentaje se calcula respecto al máximo de minutos jugados en cada liga.
        """
        if self.df_processed.empty:
            return pd.DataFrame()
            
        df = self.df_processed.copy()
        
        # Calcular el máximo de minutos por liga
        max_mins_by_league = df.groupby("league")["minutes"].transform("max")
        min_minutes_threshold = max_mins_by_league * (min_minutes_pct / 100.0)
        
        # Filtrar
        df_filtered = df[df["minutes"] >= min_minutes_threshold].copy()
        return df_filtered

    def compute_similarity(self, df_filtered: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Escala las características y convierte la distancia euclídea en una
        puntuación de similitud entre 0 y 1.
        """
        if df_filtered.empty or len(df_filtered) < 2:
            return df_filtered, np.array([])
            
        # Extraer matriz de características
        X = df_filtered[self.feature_cols].values
        
        # Escalar
        X_scaled = self.scaler.fit_transform(X)
        
        # La distancia máxima teórica entre dos vectores normalizados es sqrt(n).
        distances = pairwise_distances(X_scaled, metric="euclidean")
        max_distance = np.sqrt(X_scaled.shape[1])
        sim_matrix = np.clip(1 - (distances / max_distance), 0, 1)
        
        return df_filtered, sim_matrix

    def find_similar_players(
        self, 
        player_name: str, 
        team_name: str, 
        min_minutes_pct: float = 20.0, 
        position_filter: Optional[str] = None,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Encuentra los jugadores más similares a un jugador dado.
        """
        # 1. Buscar al jugador objetivo en el dataset COMPLETO procesado (sin filtrar por minutos)
        df_all = self.df_processed.copy()
        
        target_idx_list = df_all[
            (df_all["player"].str.lower() == player_name.lower()) & 
            (df_all["team"].str.lower() == team_name.lower())
        ].index
        
        if len(target_idx_list) == 0:
            target_idx_list = df_all[df_all["player"].str.lower() == player_name.lower()].index
            
        if len(target_idx_list) == 0:
            raise ValueError(f"Jugador '{player_name}' no encontrado en la base de datos.")
            
        target_idx_all = target_idx_list[0]
        target_player = df_all.loc[target_idx_all]
        
        # 2. Obtener el resto de jugadores que sí cumplen el filtro de minutos
        df_filtered_others = self.get_filtered_dataset(min_minutes_pct)
        
        # 3. Asegurarse de que el jugador objetivo esté incluido en el dataset para el cálculo de similitud
        # Concatenar el jugador objetivo con los otros que cumplen el filtro
        target_df = pd.DataFrame([target_player])
        
        # Si el jugador objetivo ya está en df_filtered_others, no lo duplicamos
        df_to_compare = pd.concat([target_df, df_filtered_others]).drop_duplicates(subset=["player", "team", "league", "season"])
        
        # Resetear índice para asegurar correspondencia con la matriz de similitud
        df_to_compare = df_to_compare.reset_index(drop=True)
        
        # Encontrar la nueva posición del jugador objetivo en el dataframe consolidado
        new_target_idx_list = df_to_compare[
            (df_to_compare["player"] == target_player["player"]) & 
            (df_to_compare["team"] == target_player["team"])
        ].index
        
        new_target_idx = new_target_idx_list[0]
        
        # 4. Calcular similitud
        df_to_compare, sim_matrix = self.compute_similarity(df_to_compare)
        if len(sim_matrix) == 0:
            return pd.DataFrame()
            
        # Obtener vector de similitudes para el jugador objetivo
        player_sims = sim_matrix[new_target_idx]
        
        # Crear DataFrame de resultados
        results = df_to_compare.copy()
        results["similarity"] = player_sims
        
        # Ordenar por similitud descendente
        results = results.sort_values(by="similarity", ascending=False)
        
        # Excluir al propio jugador buscado
        results = results[
            ~((results["player"] == target_player["player"]) & 
              (results["team"] == target_player["team"]))
        ]
        
        # 5. Aplicar filtro de posición interactivo si se requiere
        if position_filter and position_filter != "Todos":
            results = results[results["clean_position"] == position_filter]
            
        # Retornar el Top N
        return results.head(top_n)

    def generate_openai_report(
        self,
        player_a: pd.Series,
        player_b: pd.Series,
        api_key: str
    ) -> str:
        """
        Llama a la API de OpenAI para generar un reporte comparativo cualitativo detallado.
        """
        if not api_key:
            return "Configura una API Key válida de OpenAI en el archivo .env para generar el reporte táctico."

        # Construir un resumen estadístico comparativo para el prompt
        # Seleccionamos algunas estadísticas clave de alto impacto
        stats_to_compare = []
        
        # Buscar métricas relevantes disponibles en las características
        key_keywords = [
            "goal", "assist", "xg", "xa", "pass_cmp", "tkl", "int", "dribble", 
            "shot", "carry", "key_pass", "prg", "clearance"
        ]
        
        seen_base_names = set()
        for col in self.feature_cols:
            col_lower = col.lower()
            # Encontrar el nombre original sin '_per90'
            base_name = col.replace("_per90", "")
            if any(kw in col_lower for kw in key_keywords) and base_name not in seen_base_names:
                val_a = player_a[col]
                val_b = player_b[col]
                # Guardar si alguno de los dos tiene valores mayores a 0
                if val_a > 0 or val_b > 0:
                    stats_to_compare.append({
                        "metric": base_name,
                        "player_a": f"{val_a:.2f}",
                        "player_b": f"{val_b:.2f}"
                    })
                    seen_base_names.add(base_name)

        stats_summary = "\n".join([
            f"- {s['metric']}: {player_a['player']} = {s['player_a']} | {player_b['player']} = {s['player_b']}"
            for s in stats_to_compare[:15] # Limitar a las 15 más relevantes para ahorrar tokens
        ])

        prompt = f"""
Eres un analista de datos de fútbol profesional y un cazatalentos (scout).
Tu tarea es realizar un informe comparativo táctico de dos jugadores basándote en sus estadísticas de rendimiento de la temporada.

Datos del Jugador A (Objetivo):
- Nombre: {player_a['player']}
- Equipo: {player_a['team']}
- Liga: {player_a['league']}
- Posición: {player_a['clean_position']} ({player_a['position']})
- Edad: {player_a.get('age', 'Desconocida')} años
- Minutos Jugados: {player_a['minutes']} mins

Datos del Jugador B (Candidato Similar):
- Nombre: {player_b['player']}
- Equipo: {player_b['team']}
- Liga: {player_b['league']}
- Posición: {player_b['clean_position']} ({player_b['position']})
- Edad: {player_b.get('age', 'Desconocida')} años
- Minutos Jugados: {player_b['minutes']} mins
- Similitud Matemática Calculada: {player_b['similarity'] * 100:.1f}%
- Método de Similitud: distancia euclídea sobre métricas normalizadas entre 0 y 1.

Estadísticas Comparativas Clave (Normalizadas por 90 minutos de juego):
{stats_summary}

Escribe un reporte premium estructurado en los siguientes puntos (redactado en español):
1. **Perfil y Estilo de Juego**: Analiza brevemente cómo juega cada uno según sus estadísticas (si es más pasador, regateador, defensivo, finalizador, etc.).
2. **Fortalezas y Similitudes Clave**: ¿En qué métricas tienen valores realmente cercanos y cuáles sostienen la puntuación de similitud?
3. **Diferencias Tácticas**: Aunque son similares, ¿en qué aspectos se diferencian según los datos? (Ej. uno toma más riesgos, el otro es más eficiente en defensa, etc.).
4. **Conclusión de Scouting**: Evalúa si el Jugador B sería un reemplazo/fichaje adecuado para el rol del Jugador A, considerando también su edad y liga.

Sé conciso, analítico y profesional. Evita generalidades y usa las estadísticas provistas para justificar tus comentarios.
"""
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Usar gpt-4o-mini por eficiencia y bajo coste
                messages=[
                    {"role": "system", "content": "Eres un scout y analista de datos de fútbol profesional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error al llamar a la API de OpenAI: {e}")
            return f"Error al generar el reporte táctico con la IA: {e}. Comprueba tu clave de API."
