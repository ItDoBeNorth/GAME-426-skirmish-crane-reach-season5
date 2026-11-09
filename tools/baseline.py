"""Season 5 baseline runner: play any agent folder against a rival and report how games end.

Usage (from the repo folder, with the repo's .venv python):
    python baseline.py MAIN_FOLDER VS_FOLDER_or_naive SEAT SEED_START N_SEEDS
Prints one JSON line per game plus a summary JSON line.
"""

import json
import sys
import time
from pathlib import Path
from statistics import mean

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from sandbox import play  # noqa: E402
from sandbox.harness.environment import resolve_parameters  # noqa: E402
from sandbox.season import load_season_settings  # noqa: E402

# MAIN_FOLDER "teacher" plays the repo's teacher.py (class Agent) instead of a manifest folder.
# Its marker path is teacher.py itself, so a rival "." (the learner in the repo root) still loads
# the learner instead of being mistaken for our own seat.
use_teacher = sys.argv[1] == "teacher"
main_root = (REPO / "teacher.py").resolve() if use_teacher else Path(sys.argv[1]).resolve()
vs = None if sys.argv[2] == "naive" else Path(sys.argv[2]).resolve()
seat = int(sys.argv[3])
start, count = int(sys.argv[4]), int(sys.argv[5])
# Optional NAME=VALUE arguments override teacher.py's module switches for this run only (for
# ablations without editing teacher.py while games run), e.g. HUNT_ENEMY_ZONE=False.
overrides = {}
for item in sys.argv[6:]:
    name, _, value = item.partition("=")
    overrides[name] = eval(value)  # noqa: S307 - local experiment tool; values like False, 3, 0.5

season = load_season_settings(REPO, play.META)
parameters = resolve_parameters(play.META, season.parameters)
play.REPO_ROOT = main_root

envs = []
original_make_env = play.make_env


def keep_env(parameters):
    env = original_make_env(parameters)
    envs.append(env)
    return env


play.make_env = keep_env

# Time every act() of agents loaded from the main folder (self-play times both sides).
act_ms = []
original_load = play.load_agent


def load_teacher():
    """A fresh teacher module per unit, like the harness gives each unit a fresh agent module."""
    import importlib

    sys.modules.pop("teacher", None)
    module = importlib.import_module("teacher")
    for name, value in overrides.items():
        if not hasattr(module, name):
            raise SystemExit(f"teacher.py has no switch {name}")
        setattr(module, name, value)
    return module.Agent()


def timed_load(root):
    if use_teacher and Path(root).resolve() == main_root:
        agent = load_teacher()
    else:
        agent = original_load(root)
    if Path(root).resolve() == main_root:
        inner = agent.act

        def act(observation, inner=inner):
            t = time.perf_counter()
            result = inner(observation)
            act_ms.append(1000 * (time.perf_counter() - t))
            return result

        agent.act = act
    return agent


play.load_agent = timed_load

own_side = "red" if seat == 0 else "blue"
other_side = "blue" if seat == 0 else "red"
games = []
for seed in range(start, start + count):
    act_ms.clear()
    score = play.run_headless(
        seed=seed,
        max_steps=None,
        seat=seat,
        vs=vs,
        parameters=parameters,
        decision_limit_ms=season.decision_limit_ms,
        game_limit_ms=season.game_limit_ms,
    )
    match = envs[-1].match
    alive = {side: [u for u in match.units.values() if u.side == side and u.hit_points > 0] for side in ("red", "blue")}
    capture = dict(match.capture_scores)
    if not alive[own_side] or not alive[other_side]:
        end = "elimination"
    elif max(capture.values()) >= match.config.capture_target:
        end = "capture"
    else:
        end = "round_cap"
    game = {
        "seed": seed,
        "seat": seat,
        "score": score,
        "end": end,
        "rounds": match.round,
        "cap_own": capture[own_side],
        "cap_opp": capture[other_side],
        "alive_own": len(alive[own_side]),
        "alive_opp": len(alive[other_side]),
        "mean_act_ms": mean(act_ms) if act_ms else None,
        "max_act_ms": max(act_ms) if act_ms else None,
    }
    games.append(game)
    print(json.dumps(game), flush=True)

ends = {}
for g in games:
    ends[g["end"]] = ends.get(g["end"], 0) + 1
summary = {
    "main": "teacher" if use_teacher else main_root.name,
    "overrides": {name: repr(value) for name, value in overrides.items()},
    "vs": "naive" if vs is None else vs.name,
    "seat": seat,
    "games": len(games),
    "mean_score": mean(g["score"] for g in games),
    "wins": sum(g["score"] > 50 for g in games),
    "draws": sum(g["score"] == 50 for g in games),
    "ends": ends,
    "mean_rounds": mean(g["rounds"] for g in games),
    "mean_cap_own": mean(g["cap_own"] for g in games),
    "mean_cap_opp": mean(g["cap_opp"] for g in games),
    "mean_alive_own": mean(g["alive_own"] for g in games),
    "mean_alive_opp": mean(g["alive_opp"] for g in games),
    "mean_act_ms": mean(g["mean_act_ms"] for g in games if g["mean_act_ms"] is not None),
    "max_act_ms": max(g["max_act_ms"] for g in games if g["max_act_ms"] is not None),
}
print("SUMMARY " + json.dumps(summary), flush=True)
