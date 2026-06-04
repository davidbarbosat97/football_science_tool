import os
import logging
import pandas as pd
import soccerdata as sd
from typing import List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseScraper:
    """Clase base para scrapers de fútbol."""
    def __init__(self, leagues: List[str], cache_dir: str):
        self.leagues = leagues
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, season: str) -> str:
        raise NotImplementedError

    def load_season_data(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        raise NotImplementedError


class FBrefScraper(BaseScraper):
    """Scraper para extraer y consolidar estadísticas de FBref."""
    STAT_TYPES = ["standard", "shooting", "passing", "defense", "possession", "misc"]
    
    def __init__(self, leagues: List[str], cache_dir: str = "data"):
        super().__init__(leagues, cache_dir)

    def get_cache_path(self, season: str) -> str:
        season_clean = season.replace("/", "-")
        return os.path.join(self.cache_dir, f"fbref_players_{season_clean}.parquet")

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = []
            for col in df.columns.values:
                parts = [str(p).strip() for p in col if str(p).strip() and not str(p).startswith('Unnamed:')]
                flat_name = "_".join(parts)
                flat_cols.append(flat_name)
            df.columns = flat_cols
        return df

    def load_season_data(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self.get_cache_path(season)
        if not force_refresh and os.path.exists(cache_path):
            logger.info(f"Cargando FBref desde caché: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.error(f"Error al leer caché parquet: {e}. Recargando...")

        logger.info(f"Descargando datos FBref para temporada {season}...")
        df_consolidated = self.scrape_season(season)
        
        if df_consolidated is not None and not df_consolidated.empty:
            df_consolidated.to_parquet(cache_path)
            logger.info(f"Datos de FBref guardados en caché: {cache_path}")
            return df_consolidated
        return pd.DataFrame()

    def scrape_season(self, season: str) -> pd.DataFrame:
        try:
            # Configurar el scraper de FBref
            fbref = sd.FBref(leagues=self.leagues, seasons=season)
        except Exception as e:
            logger.error(f"Error al inicializar sd.FBref: {e}")
            raise ConnectionError(f"No se pudo conectar a FBref (posible bloqueo/403): {e}")

        consolidated_df = None
        common_cols_to_drop = [
            "Demographics_Nation", "Demographics_Pos", "Demographics_Age", "Demographics_Born",
            "Playing Time_MP", "Playing Time_Starts", "Playing Time_Min", "Playing Time_90s"
        ]

        for stat_type in self.STAT_TYPES:
            logger.info(f"FBref: Scrapeando '{stat_type}'...")
            try:
                df = fbref.read_player_season_stats(stat_type=stat_type)
                if df is None or df.empty:
                    continue
                df_flat = self._flatten_columns(df)
                
                if consolidated_df is None:
                    consolidated_df = df_flat
                else:
                    cols_to_keep = [col for col in df_flat.columns if col not in common_cols_to_drop]
                    consolidated_df = consolidated_df.merge(
                        df_flat[cols_to_keep], left_index=True, right_index=True, how="outer"
                    )
            except Exception as e:
                logger.error(f"Error FBref en '{stat_type}': {e}")
                if consolidated_df is None:
                    raise e
                continue

        if consolidated_df is not None and not consolidated_df.empty:
            consolidated_df = consolidated_df.reset_index()
            consolidated_df = consolidated_df.drop_duplicates()
            # Mapear columnas comunes a nombres unificados
            consolidated_df = consolidated_df.rename(columns={
                "Demographics_Pos": "position",
                "Demographics_Age": "age",
                "Demographics_Nation": "nation",
                "Playing Time_Min": "minutes",
                "Playing Time_MP": "matches"
            })
            return consolidated_df
        return pd.DataFrame()


class UnderstatScraper(BaseScraper):
    """Scraper para extraer estadísticas de Understat (alternativa robusta sin bloqueos)."""
    def __init__(self, leagues: List[str], cache_dir: str = "data"):
        super().__init__(leagues, cache_dir)

    def get_cache_path(self, season: str) -> str:
        # Understat usa el año de inicio (ej. "2025" para "2025-26")
        season_year = season.split("-")[0] if "-" in season else season
        return os.path.join(self.cache_dir, f"understat_players_{season_year}.parquet")

    def load_season_data(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self.get_cache_path(season)
        if not force_refresh and os.path.exists(cache_path):
            logger.info(f"Cargando Understat desde caché: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.error(f"Error al leer caché parquet: {e}. Recargando...")

        logger.info(f"Descargando datos Understat para temporada {season}...")
        df = self.scrape_season(season)
        
        if df is not None and not df.empty:
            df.to_parquet(cache_path)
            logger.info(f"Datos de Understat guardados en caché: {cache_path}")
            return df
        return pd.DataFrame()

    def scrape_season(self, season: str) -> pd.DataFrame:
        # Traducir temporada de formato "2025-26" o "2025-2026" a "2025"
        season_year = season.split("-")[0] if "-" in season else season
        try:
            understat = sd.Understat(leagues=self.leagues, seasons=int(season_year))
            df = understat.read_player_season_stats()
        except Exception as e:
            logger.error(f"Error al extraer de Understat: {e}")
            return pd.DataFrame()

        if df is not None and not df.empty:
            df = df.reset_index()
            df = df.drop_duplicates()
            # Mapear columnas para ser compatibles con las columnas unificadas
            # Understat ya tiene position, minutes, matches
            # Limpiar nombre de columnas de metadatos de Understat
            if "season" in df.columns:
                # La columna 'season' puede ser un entero en Understat, homogeneizar a string
                df["season"] = df["season"].astype(str)
            return df
        return pd.DataFrame()


def get_scraper(provider: str, leagues: List[str], cache_dir: str = "data") -> BaseScraper:
    """Factory para obtener la instancia del scraper adecuado."""
    if provider.lower() == "fbref":
        return FBrefScraper(leagues, cache_dir)
    elif provider.lower() == "understat":
        return UnderstatScraper(leagues, cache_dir)
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")
