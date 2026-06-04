import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

class PlayerRecommender:
    """
    Motor de similitud y recomendación de futbolistas.
    Preprocesa los datos, calcula métricas por 90 minutos y aplica distancia euclídea.
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
    NON_FEATURE_PREFIXES = (
        "rk", "rank", "nation", "pos", "comp", "age", "born", "season",
        "player", "squad", "team", "league", "mp", "matches", "starts",
        "min", "minutes", "90s",
    )
    MIN_FEATURE_COVERAGE = 0.05
    MIN_POSITION_FEATURE_COVERAGE = 0.5
    RATE_METRIC_NAMES = {
        "ppm", "dist", "avglen", "avgdist", "on-off",
        "g/sh", "g/sot", "npxg/sh", "mn/mp", "mn/start", "mn/sub",
        "psxg/sot",
    }

    def __init__(self, df: pd.DataFrame):
        self.df_raw = df.copy()
        self.df_processed = pd.DataFrame()
        self.feature_cols = []
        self.feature_position_coverage = {}
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

        # 4. Seleccionar estadísticas con cobertura y variabilidad suficientes.
        numeric_stat_cols = self._select_feature_columns(df)
        
        # 5. Convertir estadísticas brutas a métricas por 90 minutos
        # Evita normalizar columnas de porcentaje (ej. que contengan "%" o "pct" o "rate")
        calculated_features = {}
        self.feature_cols = []
        
        for col in numeric_stat_cols:
            col_lower = col.lower()
            is_percentage = self._is_rate_metric(col)
            is_average = "avg" in col_lower or "average" in col_lower
            
            if is_percentage or is_average:
                # Mantener porcentajes y promedios directamente
                p90_col_name = col
            else:
                # Calcular por 90 minutos
                p90_col_name = f"{col}_per90"
                calculated_features[p90_col_name] = (
                    df[col] / (df["minutes"] / 90.0)
                ).fillna(0)
                
            self.feature_cols.append(p90_col_name)
            self.feature_position_coverage[p90_col_name] = (
                df.assign(_available=df[col].notna())
                .groupby("clean_position")["_available"]
                .mean()
                .to_dict()
            )

        df_p90 = df.copy()
        if calculated_features:
            df_p90 = pd.concat(
                [df_p90, pd.DataFrame(calculated_features, index=df.index)], axis=1
            )
        # Reemplazar valores infinitos (por división por cero) por 0
        df_p90[self.feature_cols] = df_p90[self.feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        
        self.df_processed = df_p90

    def _select_feature_columns(self, df: pd.DataFrame):
        exclude = set(self.METADATA_COLS + ["clean_position"])
        selected = []
        seen_value_signatures = set()
        for column in df.columns:
            if column in exclude or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            if self._is_administrative_column(column):
                continue
            if self._has_explicit_rate_version(column, df.columns):
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().mean() < self.MIN_FEATURE_COVERAGE:
                continue
            if values.nunique(dropna=True) <= 1:
                continue
            signature = pd.util.hash_pandas_object(values, index=False).values.tobytes()
            if signature in seen_value_signatures:
                continue
            seen_value_signatures.add(signature)
            selected.append(column)
        return selected

    @staticmethod
    def _has_explicit_rate_version(column: str, all_columns) -> bool:
        candidates = {f"{column}/90", f"{column}90", f"{column}_per90"}
        return any(candidate in all_columns for candidate in candidates)

    def _is_administrative_column(self, column: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        return any(
            normalized == prefix or normalized.startswith(f"{prefix}_stats_")
            for prefix in self.NON_FEATURE_PREFIXES
        )

    @staticmethod
    def _is_rate_metric(column: str) -> bool:
        normalized = column.lower()
        return (
            normalized in PlayerRecommender.RATE_METRIC_NAMES
            or "%" in column
            or "pct" in normalized
            or "rate" in normalized
            or "percent" in normalized
            or "/90" in normalized
            or "per90" in normalized
            or "per_90" in normalized
            or normalized.endswith("90")
        )

    @staticmethod
    def feature_label(column: str) -> str:
        label = column.replace("_per90", "").replace("_", " ").strip()
        return re.sub(r"\s+", " ", label)

    def get_features_for_position(self, position: Optional[str]) -> list:
        if not position:
            return self.feature_cols.copy()
        selected = [
            feature
            for feature in self.feature_cols
            if self.feature_position_coverage.get(feature, {}).get(position, 0)
            >= self.MIN_POSITION_FEATURE_COVERAGE
        ]
        return selected or self.feature_cols.copy()

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

    def compute_target_similarity(
        self,
        df_filtered: pd.DataFrame,
        target_index: int,
        feature_cols: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Calculates similarity from one target to all players in O(N) memory."""
        if df_filtered.empty or len(df_filtered) < 2:
            return df_filtered, np.array([])

        active_features = feature_cols or self.feature_cols
        X = df_filtered[active_features].values
        X_scaled = self.scaler.fit_transform(X)
        distances = pairwise_distances(
            X_scaled[target_index].reshape(1, -1),
            X_scaled,
            metric="euclidean",
        )[0]
        max_distance = np.sqrt(X_scaled.shape[1])
        return df_filtered, np.clip(1 - (distances / max_distance), 0, 1)

    def find_similar_players(
        self, 
        player_name: str, 
        team_name: str, 
        league_name: Optional[str] = None,
        season: Optional[str] = None,
        min_minutes_pct: float = 20.0, 
        position_filter: Optional[str] = None,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Encuentra los jugadores más similares a un jugador dado.
        """
        # 1. Buscar al jugador objetivo en el dataset COMPLETO procesado (sin filtrar por minutos)
        df_all = self.df_processed.copy()
        
        target_mask = (
            (df_all["player"].str.lower() == player_name.lower()) & 
            (df_all["team"].str.lower() == team_name.lower())
        )
        if league_name is not None:
            target_mask &= df_all["league"].astype(str) == str(league_name)
        if season is not None:
            target_mask &= df_all["season"].astype(str) == str(season)
        target_idx_list = df_all[target_mask].index
        
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
            (df_to_compare["team"] == target_player["team"]) &
            (df_to_compare["league"] == target_player["league"]) &
            (df_to_compare["season"] == target_player["season"])
        ].index
        
        new_target_idx = new_target_idx_list[0]
        
        # 4. Calcular similitud
        active_features = self.get_features_for_position(target_player["clean_position"])
        df_to_compare, player_sims = self.compute_target_similarity(
            df_to_compare, new_target_idx, feature_cols=active_features
        )
        if len(player_sims) == 0:
            return pd.DataFrame()
        
        # Crear DataFrame de resultados
        results = df_to_compare.copy()
        results["similarity"] = player_sims
        
        # Ordenar por similitud descendente
        results = results.sort_values(by="similarity", ascending=False)
        
        # Excluir al propio jugador buscado
        results = results[
            ~((results["player"] == target_player["player"]) & 
              (results["team"] == target_player["team"]) &
              (results["league"] == target_player["league"]) &
              (results["season"] == target_player["season"]))
        ]
        
        # 5. Aplicar filtro de posición interactivo si se requiere
        if position_filter and position_filter != "Todos":
            results = results[results["clean_position"] == position_filter]
            
        # Retornar el Top N
        return results.head(top_n)
