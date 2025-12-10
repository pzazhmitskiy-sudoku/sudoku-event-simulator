import random
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gradio as gr

# ===================== 0. Константы и конфиги =====================

DEFAULT_EVENT_PLAYERS = 50  # всего игроков в событии по умолчанию

event_difficulty_progression = {
    "bounds": {"min_tier": 1, "max_tier": 5},
    "initial_events": {
        "enabled": True,
        "count": 3,
        "fixed_difficulty_tier": 2,
    },
    "constraints": {
        "allowed_after_previous_delta": {
            "-2": ["0", "1"],
            "-1": ["0", "1"],
            "0":  ["-1", "0", "1"],
            "1":  ["-1", "0"],
            "2":  ["-1", "0"],
        }
    },
    "types": {
        "weekly_tournament": {
            "place_rules": [
                {"place_percent_min": 0, "place_percent_max": 1, "base_difficulty_delta": 1},
                {"place_percent_min": 1, "place_percent_max": 10, "base_difficulty_delta": 1},
                {"place_percent_min": 10, "place_percent_max": 85, "base_difficulty_delta": 0},
                {"place_percent_min": 85, "place_percent_max": 95, "base_difficulty_delta": -1},
                {"place_percent_min": 95, "place_percent_max": 100, "base_difficulty_delta": -1},
            ]
        }
    },
}

activity_profiles = {
    1: (1, 3),
    2: (2, 5),
    3: (3, 7),
    4: (5, 10),
    5: (7, 15),
}

# ===================== 1. Модель игрока =====================

@dataclass
class Player:
    id: int
    activity_tier: int
    current_difficulty_tier: int = 2
    events_played: int = 0
    last_delta: int = 0
    non_prize_streak: int = 0

    history_difficulty: List[int] = field(default_factory=list)
    history_place_percent: List[float] = field(default_factory=list)
    history_place: List[int] = field(default_factory=list)
    history_levels: List[int] = field(default_factory=list)

# ===================== 2. Логика симуляции =====================

def simulate_levels_completed(player: Player, event_tier: int) -> int:
    base_min, base_max = activity_profiles[player.activity_tier]
    base_levels = random.randint(base_min, base_max)

    difficulty_penalty = 1.0 - 0.05 * (event_tier - 3)
    difficulty_penalty = max(0.7, min(1.1, difficulty_penalty))

    levels = int(base_levels * difficulty_penalty)
    return max(0, levels)


def compute_place_and_percent(player_score: int, competitors_scores: List[int]) -> Tuple[int, float]:
    all_scores = competitors_scores + [player_score]
    all_scores_sorted = sorted(all_scores, reverse=True)
    place = all_scores_sorted.index(player_score) + 1
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
    return 0


def apply_constraints(last_delta: int, base_delta: int) -> int:
    constraints = event_difficulty_progression["constraints"]["allowed_after_previous_delta"]
    allowed = constraints[str(last_delta)]
    if str(base_delta) in allowed:
        return base_delta
    if "0" in allowed:
        return 0
    return int(allowed[0])


def get_next_difficulty_tier(player: Player, final_delta: int) -> int:
    bounds = event_difficulty_progression["bounds"]
    next_tier = player.current_difficulty_tier + final_delta
    return max(bounds["min_tier"], min(bounds["max_tier"], next_tier))


def simulate_event_for_player(
    player: Player,
    event_type: str = "weekly_tournament",
    competitors_count: int = DEFAULT_EVENT_PLAYERS - 1,
):
    cfg_init = event_difficulty_progression["initial_events"]

    if cfg_init["enabled"] and player.events_played < cfg_init["count"]:
        event_tier = cfg_init["fixed_difficulty_tier"]
    else:
        event_tier = player.current_difficulty_tier

    player_levels = simulate_levels_completed(player, event_tier)

    competitors_scores = []
    for _ in range(competitors_count):
        bot_activity = random.randint(1, 5)
        fake_bot = Player(id=-1, activity_tier=bot_activity, current_difficulty_tier=event_tier)
        competitors_scores.append(simulate_levels_completed(fake_bot, event_tier))

    place, place_percent = compute_place_and_percent(player_levels, competitors_scores)

    if place <= 5:
        player.non_prize_streak = 0
    else:
        player.non_prize_streak += 1

    if cfg_init["enabled"] and player.events_played < cfg_init["count"]:
        base_delta = 0
    else:
        base_delta = find_base_delta_for_place(event_type, place_percent)
        if player.non_prize_streak >= 3 and base_delta >= 0:
            base_delta = -1
            player.non_prize_streak = 0

    final_delta = apply_constraints(player.last_delta, base_delta)
    next_tier = get_next_difficulty_tier(player, final_delta)

    player.history_difficulty.append(event_tier)
    player.history_place_percent.append(place_percent)
    player.history_place.append(place)
    player.history_levels.append(player_levels)

    player.current_difficulty_tier = next_tier
    player.last_delta = final_delta
    player.events_played += 1


def simulate_single_player(
    activity_tier: int,
    start_difficulty: int,
    num_events: int,
    players_per_event: int,
    seed: int = 42,
) -> Player:
    random.seed(seed)
    p = Player(id=0, activity_tier=activity_tier, current_difficulty_tier=start_difficulty)
    competitors_count = max(1, players_per_event - 1)
    for _ in range(num_events):
        simulate_event_for_player(p, competitors_count=competitors_count)
    return p


def run_simulation_many(
    num_players: int,
    num_events: int,
    players_per_event: int,
    seed: int = 42,
):
    random.seed(seed)
    players: List[Player] = []
    for i in range(num_players):
        activity = random.randint(1, 5)
        p = Player(id=i, activity_tier=activity)
        players.append(p)

    competitors_count = max(1, players_per_event - 1)

    for _ in range(num_events):
        for p in players:
            simulate_event_for_player(p, competitors_count=competitors_count)

    rows = []
    for p in players:
        for idx, tier in enumerate(p.history_difficulty):
            rows.append({
                "player_id": p.id,
                "activity_tier": p.activity_tier,
                "event_index": idx + 1,
                "difficulty_tier": tier,
                "place_percent": p.history_place_percent[idx],
                "place": p.history_place[idx],
                "levels_completed": p.history_levels[idx],
            })
    df = pd.DataFrame(rows)
    return players, df

# ===================== 3. Plot helpers =====================

def make_line_plot(x, y, title, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Event index")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig

# ===================== 4. Gradio wrappers =====================

def gr_single_player(activity_tier, start_difficulty, num_events, players_per_event, seed):
    activity_tier = int(activity_tier)
    start_difficulty = int(start_difficulty)
    num_events = int(num_events)
    players_per_event = int(players_per_event)
    seed = int(seed)

    player = simulate_single_player(
        activity_tier=activity_tier,
        start_difficulty=start_difficulty,
        num_events=num_events,
        players_per_event=players_per_event,
        seed=seed,
    )

    df = pd.DataFrame({
        "event_index": np.arange(1, len(player.history_difficulty) + 1),
        "difficulty_tier": player.history_difficulty,
        "place": player.history_place,
        "place_percent": player.history_place_percent,
        "levels_completed": player.history_levels,
    })

    diff_fig = make_line_plot(df["event_index"], df["difficulty_tier"],
                              "Difficulty tier per event", "Difficulty tier")
    place_fig = make_line_plot(df["event_index"], df["place"],
                               "Place per event", "Place (1 = best)")
    levels_fig = make_line_plot(df["event_index"], df["levels_completed"],
                                "Levels completed per event", "Levels")

    summary = {
        "Final difficulty tier": int(player.current_difficulty_tier),
        "Avg difficulty tier": float(df["difficulty_tier"].mean()),
        "Avg place": float(df["place"].mean()),
        "Avg levels per event": float(df["levels_completed"].mean()),
    }

    return summary, diff_fig, place_fig, levels_fig, df


def gr_many_players(num_players, num_events, players_per_event, seed):
    num_players = int(num_players)
    num_events = int(num_events)
    players_per_event = int(players_per_event)
    seed = int(seed)

    players, df = run_simulation_many(
        num_players=num_players,
        num_events=num_events,
        players_per_event=players_per_event,
        seed=seed,
    )

    avg_by_event_and_activity = (
        df.groupby(["event_index", "activity_tier"])["difficulty_tier"]
        .mean()
        .reset_index()
    )

    place_summary = (
        df.groupby("activity_tier")["place"]
        .describe()[["mean", "min", "max"]]
        .rename(columns={"mean": "avg_place"})
    )

    return avg_by_event_and_activity, place_summary, df.head(300)

# ===================== 5. Gradio UI =====================

with gr.Blocks(title="Sudoku Event Difficulty Simulation (Gradio)") as demo:
    gr.Markdown("# Симуляция сложности событий\nМодель адаптации сложности по активности и результатам игрока.")

    with gr.Tab("Один игрок"):
        with gr.Row():
            activity_slider = gr.Slider(1, 5, value=3, step=1, label="Activity tier")
            start_diff_slider = gr.Slider(1, 5, value=2, step=1, label="Start difficulty tier")
        with gr.Row():
            events_slider = gr.Slider(5, 100, value=30, step=5, label="Number of events")
            players_slider = gr.Slider(10, 200, value=DEFAULT_EVENT_PLAYERS, step=5, label="Players per event")
        seed_box = gr.Number(value=42, precision=0, label="Random seed")

        run_button_1 = gr.Button("Запустить симуляцию")

        summary_out = gr.JSON(label="Резюме")
        diff_plot_out = gr.Plot(label="Динамика сложности")
        place_plot_out = gr.Plot(label="Динамика места")
        levels_plot_out = gr.Plot(label="Динамика уровней")
        df_out = gr.Dataframe(label="История игрока")

        run_button_1.click(
            gr_single_player,
            inputs=[activity_slider, start_diff_slider, events_slider, players_slider, seed_box],
            outputs=[summary_out, diff_plot_out, place_plot_out, levels_plot_out, df_out],
        )

    with gr.Tab("Группа игроков"):
        num_players_slider = gr.Slider(20, 500, value=200, step=20, label="Number of players")
        events_many_slider = gr.Slider(5, 100, value=30, step=5, label="Number of events")
        players_many_slider = gr.Slider(10, 200, value=DEFAULT_EVENT_PLAYERS, step=5, label="Players per event")
        seed_many_box = gr.Number(value=42, precision=0, label="Random seed")

        run_button_2 = gr.Button("Запустить массовую симуляцию")

        avg_df_out = gr.Dataframe(label="Avg difficulty by event & activity")
        place_summary_out = gr.Dataframe(label="Place summary by activity_tier")
        head_df_out = gr.Dataframe(label="Sample raw data (head)")

        run_button_2.click(
            gr_many_players,
            inputs=[num_players_slider, events_many_slider, players_many_slider, seed_many_box],
            outputs=[avg_df_out, place_summary_out, head_df_out],
        )

if __name__ == "__main__":
    demo.launch()
