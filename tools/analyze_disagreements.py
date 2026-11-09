"""Group the learner's tile disagreements with the teacher on the validation seeds.

Usage (from the repo folder, with the repo's .venv python):
    python tools/analyze_disagreements.py MODEL.pt DATA.jsonl [DATA2.jsonl ...] [--validation-seeds 8 9]
For each unit type it splits validation decisions by "enemies in sight", reports tile agreement,
how often teacher / learner stood still, and how far apart the two chosen tiles are (a near miss of
1 tile is often an equally good move; a far miss is a different plan).
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from encoder import MOVE_FEATURES  # noqa: E402
from imitation.train import HEADS, predict  # noqa: E402
from imitation.train_tools import load_rows  # noqa: E402
from imitation_model import load_policy  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("data", nargs="+", type=Path)
parser.add_argument("--validation-seeds", nargs="+", type=int, default=[8, 9])
args = parser.parse_args()

rows = [row for row in load_rows(args.data, HEADS) if row["seed"] in args.validation_seeds]
model = load_policy(args.model)
picks = predict(model, rows)

index = {name: i for i, name in enumerate(MOVE_FEATURES)}
SIDE = 25  # end_q / end_r are offsets divided by the grid width (25 at field_extent 12)


def end_of(row, option):
    values = row["move_options"][option]
    return round(values[index["end_q"]] * SIDE), round(values[index["end_r"]] * SIDE)


def hex_distance(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


stats = defaultdict(Counter)
for row, pick in zip(rows, picks):
    if len(row["move_options"]) < 2:
        continue
    teacher, learner = row["move_choice"], pick["move"]
    sees = row["move_options"][0][index["visible_enemies"]] > 0
    key = (row["unit_type"], "enemies in sight" if sees else "no enemy in sight")
    s = stats[key]
    s["decisions"] += 1
    s["agree"] += teacher == learner
    teacher_end, learner_end = end_of(row, teacher), end_of(row, learner)
    s["teacher_stays"] += teacher_end == (0, 0)
    s["learner_stays"] += learner_end == (0, 0)
    if teacher != learner:
        gap = hex_distance(teacher_end, learner_end)
        s["miss_1_tile"] += gap == 1
        s["miss_2_tiles"] += gap == 2
        s["miss_3plus"] += gap >= 3
        s["miss_teacher_stayed"] += teacher_end == (0, 0)
        s["miss_learner_stayed"] += learner_end == (0, 0)

print(f"validation seeds {args.validation_seeds}: {len(rows)} decisions")
header = "unit / situation                     n   agree  t-stays l-stays | misses: 1 tile 2 tiles 3+   t-stayed l-stayed"
print(header)
for key in sorted(stats):
    s = stats[key]
    n, misses = s["decisions"], s["decisions"] - s["agree"]
    pct = lambda count, total: f"{100 * count / total:5.1f}%" if total else "   - "  # noqa: E731
    print(
        f"{key[0]:8s} {key[1]:20s} {n:6d}  {pct(s['agree'], n)}  {pct(s['teacher_stays'], n)} {pct(s['learner_stays'], n)} |"
        f"        {pct(s['miss_1_tile'], misses)} {pct(s['miss_2_tiles'], misses)} {pct(s['miss_3plus'], misses)}"
        f"  {pct(s['miss_teacher_stayed'], misses)} {pct(s['miss_learner_stayed'], misses)}"
    )
