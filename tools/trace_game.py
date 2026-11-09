"""Trace one game round by round: who is alive, where, who holds each zone, and who acts.

Usage (from the repo folder, with the repo's .venv python):
    python tools/trace_game.py MAIN VS SEAT SEED [EVERY]
MAIN / VS work as in tools/baseline.py ("teacher" = teacher.py, "naive", or an agent folder; a VS of
"." is the learner in the repo root). EVERY (default 10) prints a snapshot every that many rounds.
"""

import importlib
import sys
from collections import Counter
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from sandbox import play  # noqa: E402
from sandbox.env.skirmish_crane import engine  # noqa: E402
from sandbox.env.skirmish_crane.hexes import distance  # noqa: E402
from sandbox.harness.environment import resolve_parameters  # noqa: E402
from sandbox.season import load_season_settings  # noqa: E402

use_teacher = sys.argv[1] == "teacher"
main_root = (REPO / "teacher.py").resolve() if use_teacher else Path(sys.argv[1]).resolve()
vs = None if sys.argv[2] == "naive" else Path(sys.argv[2]).resolve()
seat, seed = int(sys.argv[3]), int(sys.argv[4])
every = int(sys.argv[5]) if len(sys.argv) > 5 else 10

season = load_season_settings(REPO, play.META)
parameters = resolve_parameters(play.META, season.parameters)
play.REPO_ROOT = main_root
original_load = play.load_agent


def load(root):
    if use_teacher and Path(root).resolve() == main_root:
        sys.modules.pop("teacher", None)
        return importlib.import_module("teacher").Agent()
    return original_load(root)


play.load_agent = load

# Snapshot every round end, before capture scoring, from inside the engine.
snapshots = []
original_advance = engine.Match._advance_after_activation


def advance(self):
    if self.current_unit_id is None:
        snapshots.append(
            {
                "round": self.round,
                "units": [(u.unit_id, u.side, u.kind, u.position, u.hit_points) for u in self.units.values()],
                "capture": dict(self.capture_scores),
                "zones": [(z.center, set(z.tiles)) for z in self.battlefield.zones],
                "history_len": len(self.history),
            }
        )
    original_advance(self)


engine.Match._advance_after_activation = advance

envs = []
original_make_env = play.make_env
play.make_env = lambda p: envs.append(original_make_env(p)) or envs[-1]

score = play.run_headless(
    seed=seed, max_steps=None, seat=seat, vs=vs, parameters=parameters,
    decision_limit_ms=season.decision_limit_ms, game_limit_ms=season.game_limit_ms,
)
match = envs[-1].match
own = "red" if seat == 0 else "blue"
other = "blue" if seat == 0 else "red"
print(f"seed {seed} seat {seat} ({own}) score {score:.1f} rounds {match.round} capture {match.capture_scores}")
zones = snapshots[0]["zones"]
print("zones:", [center for center, _ in zones])


def describe(snapshot, start_index, end_index):
    units = snapshot["units"]
    history = match.history[start_index:end_index]
    moved = Counter(a.unit_id for a in history if a.start != a.end)
    struck = Counter(a.unit_id for a in history if a.strike is not None)
    acted = Counter(a.unit_id for a in history)
    holders = []
    for center, tiles in zones:
        sides = {side for _, side, _, pos, _ in units if pos in tiles}
        holders.append(f"{center}:{'/'.join(sorted(sides)) or '-'}")
    print(f"--- round {snapshot['round']}  capture {snapshot['capture']}  zones {' '.join(holders)}")
    for side in (own, other):
        rows = []
        for unit_id, unit_side, kind, pos, hp in sorted(units):
            if unit_side != side:
                continue
            enemies = [p for _, s, _, p, _ in units if s != side]
            nearest = min((distance(pos, p) for p in enemies), default=None)
            zone_mark = next((i for i, (_, tiles) in enumerate(zones) if pos in tiles), None)
            rows.append(
                f"{kind[0]}{unit_id.split('_')[-1]}@{pos[0]},{pos[1]} hp{hp} near{nearest}"
                + (f" Z{zone_mark}" if zone_mark is not None else "")
                + f" mv{moved[unit_id]}/{acted[unit_id]} hit{struck[unit_id]}"
            )
        print(f"  {'OURS ' if side == own else 'THEIRS'} ({len(rows)}): " + "; ".join(rows))


previous = 0
for snapshot in snapshots:
    if snapshot["round"] % every == 0 or snapshot is snapshots[-1]:
        describe(snapshot, previous, snapshot["history_len"])
        previous = snapshot["history_len"]
