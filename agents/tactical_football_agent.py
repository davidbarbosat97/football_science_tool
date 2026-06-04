import logging
import re
from typing import Iterable, Optional

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


class TacticalFootballAgent:
    """Genera informes tácticos comparativos a partir de datos de jugadores."""

    KEY_METRIC_KEYWORDS = (
        "goal", "assist", "xg", "xa", "pass_cmp", "tkl", "int", "dribble",
        "shot", "carry", "key_pass", "prg", "clearance", "gls", "sot",
        "sca", "gca", "kp", "clr", "recov",
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        client: Optional[OpenAI] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.client = client

    def generate_report(
        self,
        player_a: pd.Series,
        player_b: pd.Series,
        feature_cols: Iterable[str],
    ) -> str:
        """Genera un reporte cualitativo comparando dos jugadores."""
        if not self.api_key and self.client is None:
            return "Configura una API Key válida de OpenAI en el archivo .env para generar el reporte táctico."

        prompt = self._build_prompt(player_a, player_b, feature_cols)

        try:
            client = self.client or OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un scout y analista de datos de fútbol profesional.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("Error al llamar a la API de OpenAI: %s", exc)
            return (
                f"Error al generar el reporte táctico con la IA: {exc}. "
                "Comprueba tu clave de API."
            )

    def _build_prompt(
        self,
        player_a: pd.Series,
        player_b: pd.Series,
        feature_cols: Iterable[str],
    ) -> str:
        stats_summary = self._build_stats_summary(player_a, player_b, feature_cols)

        return f"""
Eres un analista de datos de fútbol profesional y un cazatalentos (scout).
Tu tarea es realizar un informe comparativo táctico de dos jugadores basándote en sus estadísticas de rendimiento de la temporada.

Datos del Jugador A (Objetivo):
- Nombre: {player_a['player']}
- Equipo: {player_a['team']}
- Liga: {player_a['league']}
- Posición: {player_a['clean_position']} ({player_a['position']})
- Edad: {player_a.get('age', 'Desconocida')} años
- Nacionalidad: {player_a.get('nation', 'Desconocida')}
- Minutos Jugados: {player_a['minutes']} mins

Datos del Jugador B (Candidato Similar):
- Nombre: {player_b['player']}
- Equipo: {player_b['team']}
- Liga: {player_b['league']}
- Posición: {player_b['clean_position']} ({player_b['position']})
- Edad: {player_b.get('age', 'Desconocida')} años
- Nacionalidad: {player_b.get('nation', 'Desconocida')}
- Minutos Jugados: {player_b['minutes']} mins
- Similitud Matemática Calculada: {player_b['similarity'] * 100:.1f}%
- Método de Similitud: distancia euclídea sobre métricas normalizadas entre 0 y 1.

Estadísticas Comparativas Clave (normalizadas por 90 minutos o tasas originales):
{stats_summary}

Escribe un reporte premium estructurado en los siguientes puntos (redactado en español):
1. **Perfil y Estilo de Juego**: Analiza brevemente cómo juega cada uno según sus estadísticas (si es más pasador, regateador, defensivo, finalizador, etc.).
2. **Fortalezas y Similitudes Clave**: ¿En qué métricas tienen valores realmente cercanos y cuáles sostienen la puntuación de similitud?
3. **Diferencias Tácticas**: Aunque son similares, ¿en qué aspectos se diferencian según los datos? (Ej. uno toma más riesgos, el otro es más eficiente en defensa, etc.).
4. **Conclusión de Scouting**: Evalúa si el Jugador B sería un reemplazo/fichaje adecuado para el rol del Jugador A, considerando también su edad y liga.

Sé conciso, analítico y profesional. Evita generalidades y usa las estadísticas provistas para justificar tus comentarios.
"""

    def _build_stats_summary(
        self,
        player_a: pd.Series,
        player_b: pd.Series,
        feature_cols: Iterable[str],
    ) -> str:
        stats_to_compare = []
        seen_base_names = set()

        for column in feature_cols:
            base_name = self._feature_label(column)
            if (
                any(keyword in column.lower() for keyword in self.KEY_METRIC_KEYWORDS)
                and base_name not in seen_base_names
                and (player_a[column] > 0 or player_b[column] > 0)
            ):
                stats_to_compare.append(
                    f"- {base_name}: {player_a['player']} = {player_a[column]:.2f} | "
                    f"{player_b['player']} = {player_b[column]:.2f}"
                )
                seen_base_names.add(base_name)

        return "\n".join(stats_to_compare[:15])

    @staticmethod
    def _feature_label(column: str) -> str:
        label = column.replace("_per90", "").replace("_", " ").strip()
        return re.sub(r"\s+", " ", label)
