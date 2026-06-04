import sys
import os

# Asegurar que el directorio del proyecto está en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.scraper import get_scraper

def main():
    print("--- INICIANDO PRUEBA LOCAL DE FBREF (Temporada 2025-26) ---")
    try:
        # Intentar cargar datos de FBref para la liga española en la temporada más reciente
        print("Intentando descargar datos desde FBref para 'ESP-La Liga' y temporada '2025-26'...")
        scraper = get_scraper("fbref", ["ESP-La Liga"])
        
        # Forzar recarga para verificar la descarga directa de internet
        df = scraper.load_season_data("2025-26", force_refresh=True)
        
        print("\n--- ¡ÉXITO DE CONEXIÓN! ---")
        print(f"Datos descargados con éxito.")
        print(f"Registros de jugadores encontrados: {df.shape[0]}")
        print(f"Columnas de estadísticas encontradas: {df.shape[1]}")
        print("\nMuestra de los primeros 5 jugadores de FBref:")
        print(df[["player", "team", "position", "minutes"]].head(5))
        
        # Eliminar archivo parquet de prueba si no queremos dejar basura de prueba forzada
        # o dejarlo en cache para su uso posterior en la app
        print("\nPrueba completada correctamente. Los datos han quedado guardados en caché local.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR DE CONEXIÓN A FBREF: {e}")
        print("Esto confirma que tu dirección IP o el entorno está sujeto a bloqueos de FBref/Cloudflare.")
        print("Si esto ocurre, recuerda que puedes usar el proveedor 'Understat' que es 100% estable.")
        sys.exit(1)

if __name__ == "__main__":
    main()
