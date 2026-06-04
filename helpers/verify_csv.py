import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_loader import DatasetValidationError, PlayerDatasetNormalizer
from src.recommender import PlayerRecommender


def main():
    parser = argparse.ArgumentParser(
        description="Valida un CSV externo y muestra las métricas detectadas."
    )
    parser.add_argument("csv_path", help="Ruta al CSV de jugadores")
    parser.add_argument("--season", default="CSV", help="Temporada por defecto")
    args = parser.parse_args()

    try:
        df = PlayerDatasetNormalizer.read_csv(args.csv_path, default_season=args.season)
        recommender = PlayerRecommender(df)
    except DatasetValidationError as exc:
        print(f"CSV no compatible: {exc}")
        sys.exit(1)

    print(f"Jugadores válidos: {len(recommender.df_processed):,}")
    print(f"Ligas detectadas: {recommender.df_processed['league'].nunique():,}")
    print(f"Métricas válidas: {len(recommender.feature_cols):,}")
    for metric in recommender.feature_cols:
        print(f"- {metric}")


if __name__ == "__main__":
    main()
