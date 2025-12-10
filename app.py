import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import gradio as gr


# ======================================================
# 1. КОНФИГ: СЕГМЕНТАЦИЯ ПО АКТИВНОСТИ
# ======================================================

activity_config: Dict[str, Any] = {
    "activity_segmentation": {
        "window_days": 7,
        "tiers": {
            "1": {"name": "very_low",   "min_levels_per_active_day": 0,  "max_levels_per_active_day": 1},
            "2": {"name": "low",        "min_levels_per_active_day": 2,  "max_levels_per_active_day": 3},
            "3": {"name": "medium_low", "min_levels_per_active_day": 4,  "max_levels_per_active_day": 5},
            "4": {"name": "medium",     "min_levels_per_active_day": 6,  "max_levels_per_active_day": 7},
            "5": {"name": "medium_high","min_levels_per_active_day": 8,  "max_levels_per_active_day": 10},
            "6": {"name": "high",       "min_levels_per_active_day": 11, "max_levels_per_active_day": 13},
            "7": {"name": "very_high",  "min_levels_per_active_day": 14, "max_levels_per_active_day": 16},
            "8": {"name": "ultra_high", "min_levels_per_active_day": 17, "max_levels_per_active_day": 19},
            "9": {"name": "extreme",    "min_levels_per_active_day": 20, "max_levels_per_active_day": 22},
            "10":{"name": "legendary",  "min_levels_per_active_day": 23, "max_levels_per_active_day": 999}
        }
    }
}

# ======================================================
# 2. КОНФИГ: СЛОЖНОСТИ СОБЫТИЙ (ТРЕБУЕМЫЕ УРОВНИ ДЛЯ 1/2/3 МЕСТА)
# ======================================================

event_difficulty_config: Dict[str, Any] = {
    "event_difficulty": {
        "tiers": {
            "1": "very_easy",
            "2": "easy",
            "3": "medium",
            "4": "hard",
            "5": "extreme"
        },
        "reference": {
            "difficulty": "medium",
            "place": "2"
        },
        "difficulty_curves": {
            "very_easy": {
                "levels_multipliers_for_places": {"1": 0.8, "2": 0.6, "3": 0.4}
            },
            "easy": {
                "levels_multipliers_for_places": {"1": 1.0, "2": 0.8, "3": 0.6}
            },
            "medium": {
                "levels_multipliers_for_places": {"1": 1.2, "2": 1.0, "3": 0.8}
            },
            "hard": {
                "levels_multipliers_for_places": {"1": 1.4, "2": 1.2, "3": 1.0}
            },
            "extreme": {
                "levels_multipliers_for_places": {"1": 1.6, "2": 1.4, "3": 1.2}
            }
        }
    }
}

# ======================================================
# 3. КОНФИГ: АДАПТАЦИЯ СЛОЖНОСТИ
# ======================================================

difficulty_adaptation_config_template: Dict[str, Any] = {
    "difficulty_adaptation": {
        "bounds": {
            "min_tier": 1,
            "max_tier": 5
        },
        "initial_events": {
            "enabled": True,
            "count": 3,
            "fixed_difficulty_tier": 2
        },
        "activity_detection": {
            "min_levels_to_be_active": 1
        },
        "constraints": {
            "no_two_up_in_a_row": True,
            "no_two_down_in_a_row": True,
            "allow_zero_anytime": True
        },
        "streak_rules": {
            "active_non_prize": {
                "enabled": True,
                "prize_place_max_by_event_type": {
                    "short": 1,
                    "long": 5
                },
                "streak_length": 3,
                "override_delta": -1,
                "no_double_decrease": True
            },
            "inactivity": {
                "enabled": True,
                "inactive_on_zero_levels": True,
                "inactive_events_limit": 2,
                "delta_if_consecutive_inactive_at_least": -1
            }
        },
        "event_types": {
            "short": {
                "max_players": 5,
                "place_shift_rules": [
                    {"place_min": 1, "place_max": 1, "delta": 1},
                    {"place_min": 2, "place_max": 2, "delta": 1},
                    {"place_min": 3, "place_max": 3, "delta": 0},
                    {"place_min": 4, "place_max": 5, "delta": -1}
                ]
            },
            "long": {
                "max_players": 50,
                "place_shift_rules": [
                    {"place_min": 1,  "place_max": 1,  "delta": 1},
                    {"place_min": 2,  "place_max": 5,  "delta": 1},
                    {"place_min": 6,  "place_max": 35, "delta": 0},
                    {"place_min": 36, "place_max": 50, "delta": -1}
                ]
            }
        }
    }
}

# ======================================================
# 4. МОДЕЛЬ ИГРОКА
# ======================================================

@dataclass
class Player:
    id: int
    min_levels_per_day: int
    max_levels_per_day: int
    activity_tier: int = 1

    current_difficulty_tier: int = 2  # 1..5
    events_played: int = 0
    last_delta: int = 0
    active_non_prize_streak: int = 0
    inactive_streak: int = 0

    history_difficulty: List[int] = field(default_factory=list)          # tier сложности по событиям
    history_place: List[int] = field(default_factory=list)               # место по событиям
    history_levels: List[int] = field(default_factory=list)              # суммарные уровни за событие
    history_delta: List[int] = field(default_factory=list)               # изменение тира по событиям
    history_activity_tier_events: List[int] = field(default_factory=list)# activity-tier на начало события
    history_event_duration_days: List[int] = field(default_factory=list) # длительность события (дни)

    daily_levels: List[int] = field(default_factory=list)                # уровни по дням
    daily_activity_tier: List[int] = field(default_factory=list)         # activity-tier по дням


# ======================================================
# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================

def clone_difficulty_adaptation_config() -> Dict[str, Any]:
    import copy
    return copy.deepcopy(difficulty_adaptation_config_template)["difficulty_adaptation"]

def classify_activity_tier_by_avg(avg_levels: float) -> int:
    tiers = activity_config["activity_segmentation"]["tiers"]
    # прямое попадание
    for tid, info in tiers.items():
        if info["min_levels_per_active_day"] <= avg_levels <= info["max_levels_per_active_day"]:
            return int(tid)
    # ближайший по центру
    best_tid = None
    best_dist = float("inf")
    for tid, info in tiers.items():
        center = (info["min_levels_per_active_day"] + info["max_levels_per_active_day"]) / 2.0
        dist = abs(avg_levels - center)
        if dist < best_dist:
            best_dist = dist
            best_tid = int(tid)
    return best_tid if best_tid is not None else 1

def compute_place(player_score: int, competitors_scores: List[int]) -> int:
    all_scores = competitors_scores + [player_score]
    all_scores_sorted = sorted(all_scores, reverse=True)
    return all_scores_sorted.index(player_score) + 1

def get_base_delta_for_place(cfg_da: Dict[str, Any], event_type: str, place: int) -> int:
    cfg = cfg_da["event_types"][event_type]
    for rule in cfg["place_shift_rules"]:
        if rule["place_min"] <= place <= rule["place_max"]:
            return rule["delta"]
    return 0

def apply_constraints(cfg_da: Dict[str, Any], last_delta: int, proposed_delta: int) -> int:
    c = cfg_da["constraints"]
    if proposed_delta == 0:
        return 0
    if proposed_delta > 0 and c["no_two_up_in_a_row"] and last_delta > 0:
        return 0
    if proposed_delta < 0 and c["no_two_down_in_a_row"] and last_delta < 0:
        return 0
    return proposed_delta

def clamp_difficulty_tier(cfg_da: Dict[str, Any], tier: int) -> int:
    b = cfg_da["bounds"]
    return max(b["min_tier"], min(b["max_tier"], tier))


def get_required_levels_for_event(difficulty_tier: int, activity_tier: int, duration_days: int) -> tuple:
    """
    Требуемые уровни для 1/2/3 места исходя из:
    - тира сложности события,
    - тира активности игрока (берём средние значения его сегмента),
    - длительности события в днях.
    required_levels = avg_levels_per_day * duration_days * multiplier
    """
    conf = event_difficulty_config["event_difficulty"]
    tiers_map = conf["tiers"]
    curves = conf["difficulty_curves"]

    difficulty_name = tiers_map.get(str(difficulty_tier), "medium")
    multipliers = curves.get(difficulty_name, curves["medium"])["levels_multipliers_for_places"]

    # берём среднее из сегмента активности (уровней в день)
    act_tiers = activity_config["activity_segmentation"]["tiers"]
    act_info = act_tiers.get(str(activity_tier), act_tiers["4"])
    avg_levels_per_day = (act_info["min_levels_per_active_day"] + act_info["max_levels_per_active_day"]) / 2.0

    base = avg_levels_per_day * max(1, duration_days)

    req1 = base * multipliers.get("1", 1.0)
    req2 = base * multipliers.get("2", 1.0)
    req3 = base * multipliers.get("3", 1.0)

    return req1, req2, req3


# ======================================================
# 6. СИМУЛЯЦИЯ ДЛЯ ОДНОГО ИГРОКА
# ======================================================

def run_simulation_for_single_player(
    min_levels_per_day: int,
    max_levels_per_day: int,
    events_count: int,
    event_type: str,
    short_event_duration_days: int,
    activity_update_period_days: int,
    active_days_indices: List[int],
    seed: int,
    initial_enabled: bool,
    initial_count: int,
    initial_fixed_tier: int,
    no_two_up_in_a_row: bool,
    no_two_down_in_a_row: bool,
    active_non_prize_streak_length: int,
    inactive_events_limit: int
) -> Player:

    cfg_da = clone_difficulty_adaptation_config()

    cfg_da["initial_events"]["enabled"] = initial_enabled
    cfg_da["initial_events"]["count"] = initial_count
    cfg_da["initial_events"]["fixed_difficulty_tier"] = initial_fixed_tier

    cfg_da["constraints"]["no_two_up_in_a_row"] = no_two_up_in_a_row
    cfg_da["constraints"]["no_two_down_in_a_row"] = no_two_down_in_a_row

    cfg_da["streak_rules"]["active_non_prize"]["streak_length"] = active_non_prize_streak_length
    cfg_da["streak_rules"]["inactivity"]["inactive_events_limit"] = inactive_events_limit

    window_days = activity_config["activity_segmentation"]["window_days"]

    random.seed(seed)

    avg_levels_init = (min_levels_per_day + max_levels_per_day) / 2.0
    activity_tier_init = classify_activity_tier_by_avg(avg_levels_init)

    p = Player(
        id=1,
        min_levels_per_day=min_levels_per_day,
        max_levels_per_day=max_levels_per_day,
        activity_tier=activity_tier_init
    )

    current_day = 0

    for ev in range(events_count):
        # длительность события
        if event_type == "short":
            duration_days = short_event_duration_days
        else:
            duration_days = 7  # для long-событий

        # tier события с учётом онбординга
        if cfg_da["initial_events"]["enabled"] and ev < cfg_da["initial_events"]["count"]:
            event_tier = cfg_da["initial_events"]["fixed_difficulty_tier"]
        else:
            event_tier = p.current_difficulty_tier

        # логируем activity-tier на начало события и длительность
        p.history_activity_tier_events.append(p.activity_tier)
        p.history_event_duration_days.append(duration_days)

        # моделируем дни события
        event_score = 0
        for offset in range(duration_days):
            day_idx = current_day + offset
            pattern_day_idx = day_idx % 14
            scheduled_active = pattern_day_idx in active_days_indices

            if scheduled_active:
                lv = random.randint(min_levels_per_day, max_levels_per_day)
            else:
                lv = 0

            p.daily_levels.append(lv)
            event_score += lv

            # обновление activity-tier раз в N дней по окну window_days
            if activity_update_period_days > 0 and (day_idx + 1) % activity_update_period_days == 0:
                if len(p.daily_levels) >= window_days:
                    recent_avg = sum(p.daily_levels[-window_days:]) / window_days
                    p.activity_tier = classify_activity_tier_by_avg(recent_avg)

            p.daily_activity_tier.append(p.activity_tier)

        current_day += duration_days

        # конкурентные результаты
        cfg_event_type = cfg_da["event_types"][event_type]
        max_players = cfg_event_type["max_players"]
        competitors_count = max_players - 1

        competitors_scores = []
        for _ in range(competitors_count):
            bot_min = random.randint(0, 15)
            bot_max = bot_min + random.randint(0, 10)
            competitors_scores.append(random.randint(bot_min, bot_max))

        place = compute_place(event_score, competitors_scores)

        # активен ли в событии (по сумме)
        min_active = cfg_da["activity_detection"]["min_levels_to_be_active"]
        is_active_event = event_score >= min_active

        # базовая дельта по месту
        if cfg_da["initial_events"]["enabled"] and ev < cfg_da["initial_events"]["count"]:
            base_delta = 0
        else:
            base_delta = get_base_delta_for_place(cfg_da, event_type, place)

        # стрик "активен, но не в призах"
        ar = cfg_da["streak_rules"]["active_non_prize"]
        if ar["enabled"]:
            prize_max = ar["prize_place_max_by_event_type"][event_type]
            if is_active_event and place > prize_max:
                p.active_non_prize_streak += 1
            else:
                p.active_non_prize_streak = 0

            if p.active_non_prize_streak >= ar["streak_length"]:
                if not (ar["no_double_decrease"] and p.last_delta < 0):
                    base_delta = ar["override_delta"]
                p.active_non_prize_streak = 0

        # стрик пропусков
        ir = cfg_da["streak_rules"]["inactivity"]
        if ir["enabled"] and ir["inactive_on_zero_levels"]:
            if event_score == 0:
                p.inactive_streak += 1
            else:
                p.inactive_streak = 0

            if p.inactive_streak >= ir["inactive_events_limit"]:
                base_delta = ir["delta_if_consecutive_inactive_at_least"]

        # ограничения
        final_delta = apply_constraints(cfg_da, p.last_delta, base_delta)
        next_tier = clamp_difficulty_tier(cfg_da, event_tier + final_delta)

        # лог по событиям
        p.history_difficulty.append(event_tier)
        p.history_place.append(place)
        p.history_levels.append(event_score)
        p.history_delta.append(final_delta)

        p.current_difficulty_tier = next_tier
        p.last_delta = final_delta
        p.events_played += 1

    return p


# ======================================================
# 7. ГРАФИКИ
# ======================================================

def make_plots(player: Player):
    events = list(range(1, len(player.history_difficulty) + 1))

    # 1) График сложности по событиям
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(events, player.history_difficulty, marker="o")
    ax1.set_title("Tier сложности по событиям")
    ax1.set_xlabel("Событие")
    ax1.set_ylabel("Tier сложности")
    ax1.grid(True)
    fig1.tight_layout()

    # 2) График мест по событиям
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(events, player.history_place, marker="o")
    ax2.invert_yaxis()
    ax2.set_title("Место игрока по событиям")
    ax2.set_xlabel("Событие")
    ax2.set_ylabel("Место (1 = лучший)")
    ax2.grid(True)
    fig2.tight_layout()

    # 3) График требуемых уровней для 1/2/3 места (с учётом длительности события)
    req1_list = []
    req2_list = []
    req3_list = []
    for diff_tier, act_tier, dur in zip(
        player.history_difficulty,
        player.history_activity_tier_events,
        player.history_event_duration_days
    ):
        r1, r2, r3 = get_required_levels_for_event(diff_tier, act_tier, dur)
        req1_list.append(r1)
        req2_list.append(r2)
        req3_list.append(r3)

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(events, req1_list, marker="o", label="1 место")
    ax3.plot(events, req2_list, marker="o", label="2 место")
    ax3.plot(events, req3_list, marker="o", label="3 место")
    ax3.set_title("Нужное количество уровней для 1/2/3 места (с учётом длительности события)")
    ax3.set_xlabel("Событие")
    ax3.set_ylabel("Уровни за событие")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    # 4) График уровней в день
    days = list(range(1, len(player.daily_levels) + 1))
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.plot(days, player.daily_levels, marker="o")
    ax4.set_title("Уровни в день")
    ax4.set_xlabel("День симуляции")
    ax4.set_ylabel("Уровни")
    ax4.grid(True)
    fig4.tight_layout()

    # 5) График activity-tier по дням
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    ax5.plot(days, player.daily_activity_tier, marker="o")
    ax5.set_title("Activity-tier по дням")
    ax5.set_xlabel("День симуляции")
    ax5.set_ylabel("Activity-tier")
    ax5.grid(True)
    fig5.tight_layout()

    return fig1, fig2, fig3, fig4, fig5


# ======================================================
# 8. ОБЁРТКА ДЛЯ GRADIO
# ======================================================

def gr_run_simulation(
    min_levels_per_day,
    max_levels_per_day,
    events_count,
    event_type,
    short_event_duration_days,
    activity_update_period_days,
    active_days_pattern,
    seed,
    initial_enabled,
    initial_count,
    initial_fixed_tier,
    no_two_up_in_a_row,
    no_two_down_in_a_row,
    active_non_prize_streak_length,
    inactive_events_limit
):
    min_levels_per_day = int(min_levels_per_day)
    max_levels_per_day = int(max_levels_per_day)
    events_count = int(events_count)
    short_event_duration_days = int(short_event_duration_days)
    activity_update_period_days = int(activity_update_period_days)
    seed = int(seed)
    initial_count = int(initial_count)
    initial_fixed_tier = int(initial_fixed_tier)
    active_non_prize_streak_length = int(active_non_prize_streak_length)
    inactive_events_limit = int(inactive_events_limit)

    # парсим паттерн активности по 14 дням
    selected = active_days_pattern or []
    active_days_indices = []
    for label in selected:
        if label.startswith("D"):
            try:
                idx = int(label[1:]) - 1
                if 0 <= idx < 14:
                    active_days_indices.append(idx)
            except ValueError:
                pass

    player = run_simulation_for_single_player(
        min_levels_per_day=min_levels_per_day,
        max_levels_per_day=max_levels_per_day,
        events_count=events_count,
        event_type=event_type,
        short_event_duration_days=short_event_duration_days,
        activity_update_period_days=activity_update_period_days,
        active_days_indices=active_days_indices,
        seed=seed,
        initial_enabled=bool(initial_enabled),
        initial_count=initial_count,
        initial_fixed_tier=initial_fixed_tier,
        no_two_up_in_a_row=bool(no_two_up_in_a_row),
        no_two_down_in_a_row=bool(no_two_down_in_a_row),
        active_non_prize_streak_length=active_non_prize_streak_length,
        inactive_events_limit=inactive_events_limit
    )

    fig1, fig2, fig3, fig4, fig5 = make_plots(player)

    total_events = len(player.history_difficulty)
    participated = sum(1 for lv in player.history_levels if lv > 0)
    avg_tier = sum(player.history_difficulty) / max(1, total_events)
    avg_place = sum(player.history_place) / max(1, total_events)

    summary = {
        "min_levels_per_day": player.min_levels_per_day,
        "max_levels_per_day": player.max_levels_per_day,
        "computed_activity_tier_final": player.activity_tier,
        "final_difficulty_tier": player.current_difficulty_tier,
        "events_total": total_events,
        "events_participated": participated,
        "avg_difficulty_tier": round(avg_tier, 2),
        "avg_place": round(avg_place, 2),
        "total_days_simulated": len(player.daily_levels)
    }

    return summary, fig1, fig2, fig3, fig4, fig5


# ======================================================
# 9. GRADIO UI
# ======================================================

with gr.Blocks(title="Sudoku Event Difficulty Simulation") as demo:
    gr.Markdown(
        "# Симуляция адаптации сложности событий\n"
        "- Диапазон уровней в активный день → система сама подбирает activity-tier.\n"
        "- Паттерн активности по дням (2 недели, повторяется).\n"
        "- Обновление activity-tier раз в N дней.\n"
        "- Требуемые уровни зависят от длительности события."
    )

    with gr.Row():
        min_levels_per_day_inp = gr.Slider(0, 30, value=4, step=1, label="Мин. уровней в день")
        max_levels_per_day_inp = gr.Slider(1, 40, value=8, step=1, label="Макс. уровней в день")
        events_count_inp = gr.Slider(10, 200, value=50, step=5, label="Количество событий")

    with gr.Row():
        event_type_inp = gr.Radio(choices=["long", "short"], value="long", label="Тип события")
        short_event_duration_inp = gr.Slider(1, 3, value=2, step=1, label="Длительность короткого события (дни)")
        activity_update_period_inp = gr.Slider(1, 14, value=7, step=1, label="Обновление activity-tier каждые N дней")

    gr.Markdown("### Паттерн активности по дням (2 недели, повторяется)")
    day_labels = [f"D{i}" for i in range(1, 15)]
    active_days_pattern_inp = gr.CheckboxGroup(
        choices=day_labels,
        value=day_labels,  # по умолчанию активен каждый день
        label="Активные дни (D1..D14)"
    )

    with gr.Row():
        seed_inp = gr.Number(value=123, precision=0, label="Seed (рандом)")

    gr.Markdown("### Онбординг")
    with gr.Row():
        initial_enabled_inp = gr.Checkbox(value=True, label="Онбординг включен")
        initial_count_inp = gr.Slider(0, 10, value=3, step=1, label="Первые N событий фиксированный tier")
        initial_fixed_tier_inp = gr.Slider(1, 5, value=2, step=1, label="Фиксированный tier в онбординге")

    gr.Markdown("### Ограничения на скачки сложности")
    with gr.Row():
        no_two_up_in_a_row_inp = gr.Checkbox(value=True, label="No 2x подряд повышение")
        no_two_down_in_a_row_inp = gr.Checkbox(value=True, label="No 2x подряд понижение")

    gr.Markdown("### Стрики")
    with gr.Row():
        active_non_prize_streak_length_inp = gr.Slider(1, 10, value=3, step=1,
                                                       label="Стрик: активен без призов (событий)")
        inactive_events_limit_inp = gr.Slider(1, 10, value=2, step=1,
                                              label="Пропусков подряд до понижения")

    run_btn = gr.Button("Запустить симуляцию")

    summary_out = gr.JSON(label="Summary")
    difficulty_plot_out = gr.Plot(label="Tier сложности (по событиям)")
    place_plot_out = gr.Plot(label="Места (по событиям)")
    req_levels_plot_out = gr.Plot(label="Нужные уровни для 1/2/3 места (учёт длительности)")
    daily_levels_plot_out = gr.Plot(label="Уровни в день")
    daily_activity_plot_out = gr.Plot(label="Activity-tier по дням")

    run_btn.click(
        gr_run_simulation,
        inputs=[
            min_levels_per_day_inp,
            max_levels_per_day_inp,
            events_count_inp,
            event_type_inp,
            short_event_duration_inp,
            activity_update_period_inp,
            active_days_pattern_inp,
            seed_inp,
            initial_enabled_inp,
            initial_count_inp,
            initial_fixed_tier_inp,
            no_two_up_in_a_row_inp,
            no_two_down_in_a_row_inp,
            active_non_prize_streak_length_inp,
            inactive_events_limit_inp
        ],
        outputs=[
            summary_out,
            difficulty_plot_out,
            place_plot_out,
            req_levels_plot_out,
            daily_levels_plot_out,
            daily_activity_plot_out
        ]
    )

if __name__ == "__main__":
    demo.launch()
