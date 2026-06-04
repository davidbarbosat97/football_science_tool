import sys
import os

# Asegurar que la raíz del proyecto está en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.scraper import get_scraper
from src.recommender import PlayerRecommender

def main():
    print("--- INICIANDO VERIFICACIÓN E2E DE LA LÓGICA ---")
    try:
        # 1. Probar el scraper con Understat ya que es el más rápido y robusto
        print("1. Cargando datos desde Understat para la liga española (ESP) en 2024 (2024-25)...")
        scraper = get_scraper("understat", ["ESP-La Liga"])
        df = scraper.load_season_data("2024-25")
        
        print(f"   Datos cargados con éxito. Registros: {df.shape[0]}, Columnas: {df.shape[1]}")
        assert not df.empty, "El DataFrame no debería estar vacío."
        
        # 2. Inicializar el motor de recomendación
        print("2. Inicializando PlayerRecommender...")
        recommender = PlayerRecommender(df)
        print(f"   Características estadísticas dinámicas identificadas ({len(recommender.feature_cols)}):")
        print(f"   Muestra de características: {recommender.feature_cols[:5]}")
        
        # 3. Buscar jugadores parecidos a un jugador conocido (por ejemplo, Robert Lewandowski o Iago Aspas)
        # Vamos a listar algunos jugadores cargados para estar seguros
        players_in_db = df["player"].tolist()
        test_player = "Robert Lewandowski"
        test_team = "Barcelona"
        
        # Comprobar si está en la base de datos, si no usar el primero que aparezca
        matching_players = df[df["player"].str.lower().str.contains("lewandowski")].to_dict('records')
        if matching_players:
            test_player = matching_players[0]["player"]
            test_team = matching_players[0]["team"]
        else:
            test_player = df.iloc[0]["player"]
            test_team = df.iloc[0]["team"]
            
        print(f"3. Buscando futbolistas similares a '{test_player}' ({test_team})...")
        similar_players = recommender.find_similar_players(
            player_name=test_player,
            team_name=test_team,
            min_minutes_pct=20,
            top_n=5
        )
        
        print("\n--- RESULTADOS (Top 5 más similares) ---")
        for i, row in enumerate(similar_players.to_dict('records'), 1):
            print(f"{i}. {row['player']} ({row['team']}) - Similitud: {row['similarity']*100:.1f}% | Posición: {row['clean_position']} | Mins: {row['minutes']}")
            
        print("\n--- VERIFICACIÓN COMPLETADA CON ÉXITO ---")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR EN LA VERIFICACIÓN: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
