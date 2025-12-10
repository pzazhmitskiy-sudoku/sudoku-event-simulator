import random
from dataclasses import dataclass, field
from typing import List, Tuple
import pandas as pd
import streamlit as st

# =========================================================
# 0. КОНСТАНТЫ
# =========================================================

DEFAULT_EVENT_PLAYERS = 50  # всего игроков в событии по умолчанию


# =========================================================
# 1. КОНФИГ: ПРОГРЕССИЯ СЛОЖНОСТИ ПО МЕСТУ В СОБЫТИИ
# =========================================================

event_difficulty_progression = {
    "bounds": {
        "min_tier": 1,
        "max_tier": 5
    },

    # Первые N событий – фиксированная сложность
    "initial_events": {
        "enabled": True,
        "count": 3,
        "fixed_difficulty_tier": 2
    },

    # Ограничения: нельзя два повышения или два понижения подряд
    "constraints": {
        "allowed_after_previous_delta": {
            "-2": ["0", "1"],
            "-1": ["0", "1"],
            "0":  ["-1", "0", "1"],
            "1":  ["-1", "0"],
            "2":  ["-1", "0"]
        }
    },

    # Правила по типу события (пока один тип – weekly_tournament)
    "types": {
        "weekly_tournament": {
            "place_rules": [
                # top 1% (почти всегда 1 место)
                {"place_percent_min": 0, "place_percent_max": 1, "base_difficulty_delta": 1},
                # 1–10%
                {"place_percent_min": 1, "place_percent_max": 10, "base_difficulty_delta": 1},
                # 10–85% – середина, сложность не меняем
                {"place_percent_min": 10, "place_percent_max": 85, "base_difficulty_delta": 0},
                # 85–95% – чуть понижаем
                {"place_percent_min": 85, "place_percent_max": 95, "base_difficulty_delta": -1},
                # 95–100% – сильно внизу, тоже понижение
                {"place_percent_min": 95, "place_percent_max": 100, "base_difficulty_delta": -1}
            ]
        }
    }
}

# =========================================================
# 2. КОНФИГ: АКТИВНОСТЬ (СКОЛЬКО УРОВНЕЙ В СОБЫТИИ)
# =========================================================

# Для каждого activity_tier задаём диапазон уровней в событии
activity_profiles = {
    1: (1, 3),   # очень мало
    2: (2, 5),
    3: (3, 7),
    4: (5, 10),
    5: (7, 15)   # очень много
}


# =========================================================
# 3. МОДЕЛЬ ИГРОКА
# =========================================================

@dataclass
class Player:
    id: int
    activity_tier: int    # 1..5 – активность
    current_difficulty_tier: int = 2
    events_played: int = 0
    last_delta: int = 0
    non_prize_streak: int = 0   # подряд событий без 1–5 места

    history_difficulty: List[int] = field(default_factory=list)
    history_place_percent: List[float] = field(default_factory=list)
    history_place: List[int] = field(default_factory=list)
    history_levels: List[int] = field(default_factory=list)


# =========================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ СИМУЛЯЦИИ
# =========================================================

def simulate_levels_completed(player: Player, event_tier: int) -> int:
    """
    Модель без скилла:
    - activity_tier задаёт базовый диапазон уровней
    - сложность события чуть модифицирует результат
    """
    base_min, base_max = activity_profiles[player.activity_tier]
    base_levels = random.randint(base_min, base_max)

    # Лёгкая зависимость от сложности события:
    # вокруг normal (tier=3) почти без изменений,
    # на extreme/hard сложность может немного "съедать" объём уровней.
    difficulty_penalty = 1.0 - 0.05 * (event_tier - 3)  # tier=3 → 1.0
    difficulty_penalty = max(0.7, min(1.1, difficulty_penalty))

    levels = int(base_levels * difficulty_penalty)
    return max(0, levels)


def compute_place_and_percent(player_score: int, competitors_scores: List[int]) -> Tuple[int, float]:
    """
    Возвращает (place, place_percent):
    - place: 1..N
    - place_percent: 0% – лидер, 100% – последний
    """
    all_scores = competitors_scores + [player_score]
    all_scores_sorted = sorted(all_scores, reverse=True)
    place = all_scores_sorted.index(player_score) + 1  # 1-based
    n = len(all_scores)
    if n <= 1:
        return 1, 0.0
    place_percent = (place - 1) / (n - 1) * 100.0
    return place, place_percent


def find_base_delta_for_place(event_type: str, place_percent: float) -> int:
    rules = event_difficulty_progression["types"][event_type]["place_rules"]
    for rule in rules:
        if rule["place_percent_min"] <= place_percent < rule["place_percent_max"]:
            return rule["base_difficulty_delta"]
    return 0  # fallback


def apply_constraints(last_delta: int, base_delta: int) -> int:
    """
    Не даём два повышения или два понижения подряд (по конфигу).
    """
    constraints = event_difficulty_progression["constraints"]["allowed_after_previous_delta"]
    allowed = constraints[str(last_delta)]
    if str(base_delta) in allowed:
        return base_delta
    # Если базовый delta запрещён, пробуем 0, иначе первый разрешённый
    if "0" in allowed:
        return 0
    return int(allowed[0])


def get_next_difficulty_tier(player: Player, final_delta: int) -> int:
    bounds = event_difficulty_progression["bounds"]
    next_tier = player.current_difficulty_tier + final_delta
    next_tier = max(bounds["min_tier"], min(bounds["max_tier"], next_tier))
    return next_tier


# =========================================================
# 5. СИМУЛЯЦИЯ ОДНОГО СОБЫТИЯ ДЛЯ ИГРОКА
# =========================================================

def simulate_event_for_player(
    player: Player,
    event_type: str = "weekly_tournament",
    competitors_count: int = DEFAULT_EVENT_PLAYERS - 1
):
    cfg_init = event_difficulty_progression["initial_events"]

    # 1) Если это одно из первых событий – используем фиксированную сложность
    if cfg_init["enabled"] and player.events_played < cfg_init["count"]:
        event_tier = cfg_init["fixed_difficulty_tier"]
    else:
        event_tier = player.current_difficulty_tier

    # 2) Сколько уровней прошёл игрок
    player_levels = simulate_levels_completed(player, event_tier)

    # 3) Симулируем соперников – тоже только с activity_tier
    competitors_scores = []
    for _ in range(competitors_count):
        bot_activity = random.randint(1, 5)
        fake_bot = Player(id=-1, activity_tier=bot_activity, current_difficulty_tier=event_tier)
        competitors_scores.append(simulate_levels_completed(fake_bot, event_tier))

    # 4) Место и процент места
    place, place_percent = compute_place_and_percent(player_levels, competitors_scores)

    # 4.1) Обновляем стрик НЕпризовых мест (1–5 место считаем призовыми)
    if place <= 5:
        player.non_prize_streak = 0
    else:
        player.non_prize_streak += 1

    # 5) Считаем base_delta по занятому месту, если не онбординг
    if cfg_init["enabled"] and player.events_played < cfg_init["count"]:
        base_delta = 0
    else:
        base_delta = find_base_delta_for_place(event_type, place_percent)

        # Доп. правило: 3 события подряд без призового места → понижаем сложность
        if player.non_prize_streak >= 3 and base_delta >= 0:
            base_delta = -1
            # можно сбросить стрик, чтобы не спамить понижения
            player.non_prize_streak = 0

    # 6) Применяем ограничения (не два апа/дауна подряд)
    final_delta = apply_constraints(player.last_delta, base_delta)

    # 7) Считаем новую сложность
    next_tier = get_next_difficulty_tier(player, final_delta)

    # 8) Логируем историю и обновляем состояние игрока
    player.history_difficulty.append(event_tier)
    player.history_place_percent.append(place_percent)
    player.history_place.append(place)
    player.history_levels.append(player_levels)

    player.current_difficulty_tier = next_tier
    player.last_delta = final_delta
    player.events_played += 1


# =========================================================
# 6. СИМУЛЯЦИЯ ДЛЯ МНОГИХ ИГРОКОВ
# =========================================================

def run_simulation_many(
    num_players: int,
    num_events: int,
    players_per_event: int,
    event_type: str = "weekly_tournament"
):
    players: List[Player] = []
    for i in range(num_players):
        activity = random.randint(1, 5)
        p = Player(id=i, activity_tier=activity)
        players.append(p)

    competitors_count = max(1, players_per_event - 1)

    for _ in range(num_events):
        for p in players:
            simulate_event_for_player(p, event_type=event_type, competitors_count=competitors_count)

    # Сводим историю в DataFrame
    rows = []
    for p in players:
        for idx, tier in enumerate(p.history_difficulty):
            rows.append({
                "player_id": p.id,
                "activity_tier": p.activity_tier,
                "event_index": idx,
                "difficulty_tier": tier,
                "place_percent": p.history_place_percent[idx],
                "place": p.history_place[idx],
                "levels_completed": p.history_levels[idx]
            })
    df = pd.DataFrame(rows)
    return players, df


def simulate_single_player(
    activity_tier: int,
    start_difficulty: int,
    num_events: int,
    players_per_event: int,
    event_type: str = "weekly_tournament"
) -> Player:
    p = Player(
        id=0,
        activity_tier=activity_tier,
        current_difficulty_tier=start_difficulty
    )
    competitors_count = max(1, players_per_event - 1)

    for _ in range(num_events):
        simulate_event_for_player(p, event_type=event_type, competitors_count=competitors_count)

    return p


# =========================================================
# 7. UI: STREAMLIT ПРИЛОЖЕНИЕ
# =========================================================

st.set_page_config(page_title="Event Difficulty Simulation", layout="wide")

st.title("Симуляция сложности событий (Sudoku / турниры / ивенты)")
st.caption("Модель с сегментацией по активности, 50 игроками и адаптацией сложности по результатам.")


# --- Sidebar: режим и параметры ---
st.sidebar.header("Настройки симуляции")

mode = st.sidebar.radio(
    "Режим",
    options=["Single player", "Many players"],
    index=0
)

players_per_event = st.sidebar.slider(
    "Количество игроков в событии",
    min_value=10,
    max_value=200,
    value=DEFAULT_EVENT_PLAYERS,
    step=5,
    help="Используется для генерации соперников (bots) для каждого игрока."
)

num_events = st.sidebar.slider(
    "Количество событий",
    min_value=5,
    max_value=200,
    value=50,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.write("Конфиг:")
st.sidebar.json({
    "initial_events": event_difficulty_progression["initial_events"],
    "place_rules": event_difficulty_progression["types"]["weekly_tournament"]["place_rules"],
    "constraints": event_difficulty_progression["constraints"]
})


# =========================================================
# 8. РЕЖИМ: SINGLE PLAYER
# =========================================================

if mode == "Single player":
    st.subheader("Режим: Один игрок")

    col1, col2, col3 = st.columns(3)
    with col1:
        activity_tier = st.slider("Activity tier", 1, 5, 3)
    with col2:
        start_difficulty = st.slider("Start difficulty tier", 1, 5, 2)
    with col3:
        events_single = st.slider("Событий для игрока", 5, 100, num_events, step=5)

    run_single = st.button("Запустить симуляцию для одного игрока")

    if run_single:
        player = simulate_single_player(
            activity_tier=activity_tier,
            start_difficulty=start_difficulty,
            num_events=events_single,
            players_per_event=players_per_event
        )

        st.success("Симуляция завершена")

        # Формируем DataFrame истории
        df_single = pd.DataFrame({
            "event_index": list(range(1, len(player.history_difficulty) + 1)),
            "difficulty_tier": player.history_difficulty,
            "place_percent": player.history_place_percent,
            "place": player.history_place,
            "levels_completed": player.history_levels
        })

        # Метрики
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Последний tier сложности", player.current_difficulty_tier)
        with c2:
            st.metric("Средний tier сложности", f"{df_single['difficulty_tier'].mean():.2f}")
        with c3:
            st.metric("Среднее место", f"{df_single['place'].mean():.1f} / {players_per_event}")
        with c4:
            st.metric("Среднее кол-во уровней", f"{df_single['levels_completed'].mean():.2f}")

        st.markdown("### Динамика сложности")
        st.line_chart(df_single.set_index("event_index")["difficulty_tier"])

        st.markdown("### Динамика места")
        st.line_chart(df_single.set_index("event_index")[["place"]])

        st.markdown("### Динамика количества уровней")
        st.line_chart(df_single.set_index("event_index")[["levels_completed"]])

        with st.expander("Сырые данные игрока"):
            st.dataframe(df_single)


# =========================================================
# 9. РЕЖИМ: MANY PLAYERS
# =========================================================

if mode == "Many players":
    st.subheader("Режим: Группа игроков")

    col1, col2 = st.columns(2)
    with col1:
        num_players = st.slider("Количество игроков", 20, 500, 200, step=20)
    with col2:
        events_many = st.slider("Событий для группы", 5, 200, num_events, step=5)

    run_many = st.button("Запустить массовую симуляцию")

    if run_many:
        players, df = run_simulation_many(
            num_players=num_players,
            num_events=events_many,
            players_per_event=players_per_event
        )

        st.success("Массовая симуляция завершена")

        # Средняя сложность по событиям для разных activity_tier
        avg_by_event_and_activity = (
            df.groupby(["event_index", "activity_tier"])["difficulty_tier"]
            .mean()
            .reset_index()
        )

        st.markdown("### Средний tier сложности по событиям (по тиру активности)")
        for act in sorted(df["activity_tier"].unique()):
            subset = avg_by_event_and_activity[avg_by_event_and_activity["activity_tier"] == act]
            st.line_chart(
                subset.set_index("event_index")[["difficulty_tier"]],
                height=250
            )
            st.caption(f"Activity tier {act}")

        st.markdown("### Распределение мест по активности")
        place_summary = (
            df.groupby("activity_tier")["place"]
            .describe()[["mean", "min", "max"]]
            .rename(columns={"mean": "avg_place"})
        )
        st.dataframe(place_summary)

        st.markdown("### Пример сырых данных")
        st.dataframe(df.head(200))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Скачать результаты (CSV)",
            data=csv,
            file_name="event_simulation_results.csv",
            mime="text/csv"
        )
