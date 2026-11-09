"""Measure the Season 4 playtest feedback on real games, per unit type.

Usage (from the repo folder, with the repo's .venv python):
    python tools/behavior_stats.py MAIN VS SEED_START N_SEEDS [NAME=VALUE ...]
MAIN / VS / overrides work as in tools/baseline.py. Plays both seats per seed, replays each game's
activation history (rebuilding every unit's position) and prints one JSON line of metrics:

- footmen "shuffle": moved without striking and came back to the tile they left last activation;
  "idle": stayed without striking although no enemy was within 6 tiles.
- archers "high ground": share of our arrows shot from a hill (+1 damage onto lower ground) or out of
  a forest (cover), and the share of archer activations that END on a hill or in a forest.
- archers "stand still": stayed and shot nothing with no enemy within 6 tiles.
- archers "push": distance from each archer (after moving) to our nearest footman and nearest enemy.
- cavalry "value": activations per unit (lifespan), damage dealt per activation, charges, and
  damage taken per activation.
- "melee exposed to archers": damage our footmen/cavalry take from enemy archers per game.
- "footmen hold the zone under fire": footman activations that stay on a zone tile without striking
  while an enemy archer that can shoot them (range 6) stands in our sight range (6 tiles).
"""

import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from sandbox import play  # noqa: E402
from sandbox.env.skirmish_crane.hexes import distance  # noqa: E402
from sandbox.harness.environment import resolve_parameters  # noqa: E402
from sandbox.season import load_season_settings  # noqa: E402

use_teacher = sys.argv[1] == "teacher"
main_root = (REPO / "teacher.py").resolve() if use_teacher else Path(sys.argv[1]).resolve()
vs = None if sys.argv[2] == "naive" else Path(sys.argv[2]).resolve()
start, count = int(sys.argv[3]), int(sys.argv[4])
overrides = {}
for item in sys.argv[5:]:
    name, _, value = item.partition("=")
    overrides[name] = eval(value)  # noqa: S307 - local experiment tool

season = load_season_settings(REPO, play.META)
parameters = resolve_parameters(play.META, season.parameters)
play.REPO_ROOT = main_root
original_load = play.load_agent


def load(root):
    if use_teacher and Path(root).resolve() == main_root:
        sys.modules.pop("teacher", None)
        module = importlib.import_module("teacher")
        for key, value in overrides.items():
            setattr(module, key, value)
        return module.Agent()
    return original_load(root)


play.load_agent = load
envs = []
original_make_env = play.make_env
play.make_env = lambda p: envs.append(original_make_env(p)) or envs[-1]

ARCHER_RANGE = 6
SIGHT = 6
totals = defaultdict(Counter)  # per our unit type
lists = defaultdict(list)
games = 0
scores = []

for seed in range(start, start + count):
    for seat in (0, 1):
        own = "red" if seat == 0 else "blue"
        score = play.run_headless(
            seed=seed, max_steps=None, seat=seat, vs=vs, parameters=parameters,
            decision_limit_ms=season.decision_limit_ms, game_limit_ms=season.game_limit_ms,
        )
        scores.append(score)
        games += 1
        match = envs[-1].match
        field = match.battlefield
        zone_tiles = {position for area in field.zones for position in area.tiles}
        history = match.history

        # Rebuild positions: before its first activation a unit stands where that activation starts.
        position = {}
        for activation in history:
            position.setdefault(activation.unit_id, activation.start)
        alive = set(position)
        last_start = {}

        for activation in history:
            unit_id = activation.unit_id
            side, kind = unit_id.split("_")[0], unit_id.split("_")[1]
            position[unit_id] = activation.end
            strike = activation.strike
            moved = activation.start != activation.end
            if strike is not None:
                target_side, target_kind = strike.target_id.split("_")[0], strike.target_id.split("_")[1]
                if side == own:
                    totals[kind]["damage_dealt"] += strike.damage
                else:
                    totals[target_kind]["damage_taken"] += strike.damage
                    totals[target_kind][f"damage_taken_from_{kind}"] += strike.damage
            if activation.killed_id is not None:
                alive.discard(activation.killed_id)
                # S5 task 10: one of ours died before its next activation, i.e. on the tile it chose.
                if activation.killed_id.split("_")[0] == own:
                    totals[activation.killed_id.split("_")[1]]["died"] += 1
            if side != own:
                continue

            t = totals[kind]
            t["activations"] += 1
            t["moved"] += moved
            t["struck"] += strike is not None
            enemies = [p for u, p in position.items() if u in alive and u.split("_")[0] != own]
            nearest_enemy = min((distance(activation.end, p) for p in enemies), default=99)
            quiet = nearest_enemy > SIGHT

            if kind == "footman":
                came_back = moved and strike is None and last_start.get(unit_id) == activation.end
                t["shuffle"] += came_back
                t["idle_quiet"] += (not moved) and strike is None and quiet
                enemy_archers = [
                    p for u, p in position.items()
                    if u in alive and u.split("_")[0] != own and u.split("_")[1] == "archer"
                ]
                under_fire = any(distance(activation.end, p) <= ARCHER_RANGE for p in enemy_archers)
                if activation.end in zone_tiles and not moved and strike is None and under_fire:
                    t["hold_zone_under_archer_fire"] += 1
            if kind == "archer":
                end_tile = field.tile_at(activation.end)
                t["end_on_hill"] += end_tile.terrain == "hill"
                t["end_in_forest"] += end_tile.feature == "forest"
                t["stand_still_quiet"] += (not moved) and strike is None and quiet
                if strike is not None:
                    t["arrows"] += 1
                    t["arrows_from_hill"] += end_tile.terrain == "hill"
                    t["arrows_from_forest"] += end_tile.feature == "forest"
                footmen = [
                    p for u, p in position.items()
                    if u in alive and u != unit_id and u.split("_")[0] == own and u.split("_")[1] == "footman"
                ]
                if footmen and enemies:
                    lists["archer_to_own_footman"].append(min(distance(activation.end, p) for p in footmen))
                    lists["archer_to_enemy"].append(nearest_enemy)
            if kind == "cavalry" and strike is not None and distance(activation.start, activation.end) >= 3:
                t["charges"] += 1
            last_start[unit_id] = activation.start

        # Lifespan: activations per cavalry / footman / archer unit of ours.
        per_unit = Counter(a.unit_id for a in history if a.unit_id.startswith(own))
        for unit_id, n in per_unit.items():
            lists[f"activations_per_{unit_id.split('_')[1]}"].append(n)


def ratio(a, b):
    return round(a / b, 3) if b else None


def mean(values):
    return round(sum(values) / len(values), 2) if values else None


f, a, c = totals["footman"], totals["archer"], totals["cavalry"]
report = {
    "main": "teacher" if use_teacher else main_root.name,
    "overrides": {k: repr(v) for k, v in overrides.items()},
    "vs": "naive" if vs is None else vs.name,
    "games": games,
    "mean_score": mean(scores),
    "footman_moved": ratio(f["moved"], f["activations"]),
    "footman_shuffle": ratio(f["shuffle"], f["activations"]),
    "footman_idle_quiet": ratio(f["idle_quiet"], f["activations"]),
    "footman_hold_zone_under_archer_fire_per_game": ratio(f["hold_zone_under_archer_fire"], games),
    "archer_arrows_per_activation": ratio(a["arrows"], a["activations"]),
    "archer_arrows_from_hill": ratio(a["arrows_from_hill"], a["arrows"]),
    "archer_arrows_from_forest": ratio(a["arrows_from_forest"], a["arrows"]),
    "archer_end_on_hill": ratio(a["end_on_hill"], a["activations"]),
    "archer_end_in_forest": ratio(a["end_in_forest"], a["activations"]),
    "archer_stand_still_quiet": ratio(a["stand_still_quiet"], a["activations"]),
    "archer_damage_per_arrow": ratio(a["damage_dealt"], a["arrows"]),
    "archer_to_own_footman": mean(lists["archer_to_own_footman"]),
    "archer_to_nearest_enemy": mean(lists["archer_to_enemy"]),
    "cavalry_activations_per_unit": mean(lists["activations_per_cavalry"]),
    "cavalry_damage_per_activation": ratio(c["damage_dealt"], c["activations"]),
    "cavalry_damage_taken_per_activation": ratio(c["damage_taken"], c["activations"]),
    "cavalry_charges_per_game": ratio(c["charges"], games),
    "damage_dealt_per_game": {k: ratio(totals[k]["damage_dealt"], games) for k in ("footman", "archer", "cavalry")},
    "melee_damage_taken_from_enemy_archers_per_game": ratio(
        f["damage_taken_from_archer"] + c["damage_taken_from_archer"], games
    ),
    "activations_per_unit": {k: mean(lists[f"activations_per_{k}"]) for k in ("footman", "archer", "cavalry")},
    # S5 task 10: deaths per activation = how often a chosen tile was the last one.
    "died_per_activation": {
        k: ratio(totals[k]["died"], totals[k]["activations"]) for k in ("footman", "archer", "cavalry")
    },
    "deaths_per_game": {k: ratio(totals[k]["died"], games) for k in ("footman", "archer", "cavalry")},
}
print(json.dumps(report))
