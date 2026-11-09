#!/usr/bin/env bash
# Evaluate teacher.py switch configurations (S5 fixes) on seeds 600-619, both seats, plus behaviour
# stats vs Banner (seeds 600-609). Usage: bash tools/teacher_ablation.sh OUTDIR "name:OVERRIDES" ...
# e.g.  bash tools/teacher_ablation.sh runs/fixes "both:" "hill_only:CAVALRY_CAUTION=False"
set -uo pipefail
OUT=$1; shift
PY=.venv/Scripts/python.exe
mkdir -p $OUT
OPPONENTS="rivals/banner rivals/s5_t5b rivals/banner_learner rivals/l_t8b"
for config in "$@"; do
  name=${config%%:*}; overrides=${config#*:}
  for seat in 0 1; do
    for vs in $OPPONENTS; do
      $PY tools/baseline.py teacher $vs $seat 600 20 $overrides > $OUT/${name}_$(basename $vs)_$seat.txt 2>&1 &
    done
  done
  $PY tools/behavior_stats.py teacher rivals/banner 600 10 $overrides > $OUT/${name}_behavior.json 2> $OUT/${name}_behavior.err &
  wait
done
$PY - "$OUT" "$@" <<'PYEOF'
import json, os, sys
out, configs = sys.argv[1], [c.split(":")[0] for c in sys.argv[2:]]
def summ(p):
    if not os.path.exists(p): return None
    for line in open(p):
        if line.startswith("SUMMARY"): return json.loads(line[8:])
print("seeds 600-619 wins/40 (mean)")
for name in configs:
    parts = []
    for v in ("banner", "s5_t5b", "banner_learner", "l_t8b"):
        s = [summ(f"{out}/{name}_{v}_{seat}.txt") for seat in (0, 1)]
        parts.append(f"{v} ERROR" if None in s else f"{v} {s[0]['wins'] + s[1]['wins']}/40 ({(s[0]['mean_score'] + s[1]['mean_score']) / 2:.1f})")
    print(f"  {name:12s} " + " | ".join(parts))
keys = ["mean_score", "archer_arrows_from_hill", "archer_arrows_from_forest", "archer_damage_per_arrow", "archer_arrows_per_activation",
        "cavalry_activations_per_unit", "cavalry_damage_per_activation", "cavalry_damage_taken_per_activation", "cavalry_charges_per_game",
        "melee_damage_taken_from_enemy_archers_per_game", "damage_dealt_per_game", "footman_idle_quiet"]
rows = {}
for name in configs:
    try: rows[name] = json.load(open(f"{out}/{name}_behavior.json"))
    except Exception as e: rows[name] = {}
print(f"{'behaviour vs Banner (600-609)':48s}" + "".join(f"{n:>26s}" for n in rows))
for k in keys:
    print(f"{k:48s}" + "".join(f"{str(r.get(k)):>26s}" for r in rows.values()))
PYEOF
