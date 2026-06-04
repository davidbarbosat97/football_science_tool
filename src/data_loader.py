import re
from typing import Dict, Optional

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be adapted to the application contract."""


class PlayerDatasetNormalizer:
    """Adapts external player datasets to the application's canonical schema."""

    COLUMN_ALIASES: Dict[str, tuple] = {
        "player": ("player", "name", "player_name", "jugador"),
        "team": ("team", "squad", "club", "equipo"),
        "league": ("league", "comp", "competition", "liga"),
        "position": ("position", "pos", "posición", "posicion"),
        "minutes": ("minutes", "min", "mins", "minutos"),
        "matches": ("matches", "mp", "apps", "appearances", "partidos"),
        "age": ("age", "edad"),
        "born": ("born", "birth_year", "año_nacimiento", "ano_nacimiento"),
        "nation": ("nation", "nationality", "nacionalidad"),
        "season": ("season", "temporada"),
    }
    REQUIRED_COLUMNS = ("player", "team", "league", "position", "minutes", "matches")

    @classmethod
    def normalize(cls, df: pd.DataFrame, default_season: str = "CSV") -> pd.DataFrame:
        if df.empty:
            raise DatasetValidationError("El archivo no contiene registros.")

        normalized = df.copy()
        normalized.columns = cls._deduplicate_columns(
            [str(column).strip() for column in normalized.columns]
        )
        lookup = {}
        for column in normalized.columns:
            # Prefer the first exact-looking column (e.g. Min over Min%).
            lookup.setdefault(cls._column_key(column), column)

        rename_map = {}
        for canonical, aliases in cls.COLUMN_ALIASES.items():
            if canonical in normalized.columns:
                continue
            source = next(
                (
                    lookup[cls._column_key(alias)]
                    for alias in aliases
                    if cls._column_key(alias) in lookup
                ),
                None,
            )
            if source:
                rename_map[source] = canonical

        normalized = normalized.rename(columns=rename_map)
        missing = [column for column in cls.REQUIRED_COLUMNS if column not in normalized.columns]
        if missing:
            raise DatasetValidationError(
                "Faltan columnas obligatorias: " + ", ".join(missing)
            )

        if "season" not in normalized.columns:
            normalized["season"] = default_season

        for column in ("player", "team", "league", "position", "season"):
            normalized[column] = normalized[column].fillna("N/A").astype(str).str.strip()

        if "age" in normalized.columns:
            normalized["age"] = normalized["age"].apply(cls._normalize_age)
        if "nation" in normalized.columns:
            normalized["nation"] = normalized["nation"].apply(cls._normalize_nation)

        normalized["league"] = normalized["league"].str.replace(
            r"^[a-z]{2,3}\s+", "", regex=True
        )
        normalized["minutes"] = pd.to_numeric(normalized["minutes"], errors="coerce")
        normalized["matches"] = pd.to_numeric(normalized["matches"], errors="coerce")
        normalized = cls._convert_numeric_like_columns(normalized)
        normalized = normalized.dropna(subset=["minutes", "matches"])
        normalized = normalized[normalized["minutes"] > 0].copy()
        normalized = normalized.drop_duplicates(
            subset=["player", "team", "league", "season"], keep="first"
        )

        if normalized.empty:
            raise DatasetValidationError(
                "No quedan jugadores con minutos válidos después de normalizar el archivo."
            )
        return normalized.reset_index(drop=True)

    @classmethod
    def read_csv(
        cls,
        source,
        default_season: str = "CSV",
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        try:
            df = pd.read_csv(source, low_memory=False, encoding=encoding)
            if len(df.columns) == 1:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, low_memory=False, encoding="latin-1")
        except Exception as exc:
            raise DatasetValidationError(f"No se pudo leer el CSV: {exc}") from exc
        return cls.normalize(df, default_season=default_season)

    @staticmethod
    def _column_key(column: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", column.lower())

    @staticmethod
    def _deduplicate_columns(columns):
        counts = {}
        result = []
        for column in columns:
            count = counts.get(column, 0)
            result.append(column if count == 0 else f"{column}_{count + 1}")
            counts[column] = count + 1
        return result

    @staticmethod
    def _normalize_age(value) -> str:
        if pd.isna(value):
            return "N/A"

        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric_value) and float(numeric_value).is_integer():
            return str(int(numeric_value))
        return str(value).strip()

    @staticmethod
    def _normalize_nation(value) -> str:
        if pd.isna(value):
            return "N/A"

        uppercase_codes = re.findall(r"\b[A-Z]{2,}\b", str(value))
        return " ".join(uppercase_codes) or "N/A"

    @staticmethod
    def _convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
        converted = df.copy()
        for column in converted.select_dtypes(include="object").columns:
            if column in {
                "player", "team", "league", "position", "season", "age", "nation"
            }:
                continue
            numeric = pd.to_numeric(converted[column], errors="coerce")
            non_null_count = converted[column].notna().sum()
            if non_null_count and numeric.notna().sum() / non_null_count >= 0.8:
                converted[column] = numeric
        return converted
