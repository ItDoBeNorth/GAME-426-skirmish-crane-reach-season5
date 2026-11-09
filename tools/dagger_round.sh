#!/usr/bin/env bash
# One DAgger round (S5 task 11): the installed learner (imitation-policy.pt) plays while teacher
# copies label every situation it reaches; retrain on the teacher's demonstrations plus these
# corrections; freeze and evaluate like tools/feature_round.sh.
# Usage, from the repo folder:
#   bash tools/dagger_round.sh NAME DEMO_DIR [EXTRA_DATA_DIR ...] -- [EXTRA_OPPONENT_FOLDER ...]
#   DEMO_DIR holds the teacher-driven banner_*.jsonl / self_*.jsonl of the same encoder version;
#   EXTRA_DATA_DIR are earlier correction rounds to keep (their corrections_*.jsonl).
# Corrections: seeds 0-19 vs Banner and vs the teacher, both seats, into runs/NAME/.
set -euo pipefail
NAME=$1; shift
DATA_DIRS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do DATA_DIRS+=("$1"); shift; done
[ $# -gt 0 ] && shift
EXTRA="$*"
PY=.venv/Scripts/python.exe
mkdir -p runs/$NAME runs/${NAME}_eval

for start in 0 5 10 15; do
  $PY -m imitation.experiment collect --driver learner --episodes 5 --start $start --opponent rivals/banner --output runs/$NAME/corrections_banner_$start.jsonl > runs/$NAME/corrections_banner_$start.log 2>&1 &
  $PY -m imitation.experiment collect --driver learner --episodes 5 --start $start --opponent teacher --output runs/$NAME/corrections_teacher_$start.jsonl > runs/$NAME/corrections_teacher_$start.log 2>&1 &
done
wait
tail -qn1 runs/$NAME/*.log

FILES=()
DEMO=${DATA_DIRS[0]}
FILES+=($DEMO/banner_*.jsonl $DEMO/self_*.jsonl)
for dir in "${DATA_DIRS[@]:1}"; do FILES+=($dir/corrections_*.jsonl); done
FILES+=(runs/$NAME/corrections_*.jsonl)
$PY -m imitation.train "${FILES[@]}" --validation-seeds 8 9 18 19 --output runs/$NAME.pt > runs/$NAME/train.log 2>&1
grep -v 'Joint policy' runs/$NAME/train.log | grep -v '^\s*$'

FOLDER=rivals/l_$NAME
mkdir -p $FOLDER
cp agent.py encoder.py imitation_model.py manifest.json $FOLDER/
for helper in routes.py enemy_memory.py; do [ -f $helper ] && cp $helper $FOLDER/; done
cp runs/$NAME.pt $FOLDER/imitation-policy.pt

OPPONENTS="rivals/banner rivals/s5_t5b rivals/banner_learner $EXTRA"
for seat in 0 1; do
  for vs in $OPPONENTS; do
    $PY tools/baseline.py $FOLDER $vs $seat 600 20 > runs/${NAME}_eval/$(basename $vs)_$seat.txt 2>&1 &
  done
done
$PY tools/behavior_stats.py $FOLDER rivals/banner 600 10 > runs/${NAME}_eval/behavior.json 2> runs/${NAME}_eval/behavior.err &
wait
$PY - "$NAME" $OPPONENTS <<'EOF'
import json, os, sys
name, opponents = sys.argv[1], sys.argv[2:]
def summ(p):
    if not os.path.exists(p): return None
    for line in open(p):
        if line.startswith("SUMMARY"): return json.loads(line[8:])
print(f"{name} on seeds 600-619, wins/40 (mean; red/blue)")
for vs in opponents:
    v = os.path.basename(vs)
    s = [summ(f"runs/{name}_eval/{v}_{seat}.txt") for seat in (0, 1)]
    print(f"  vs {v}: ERROR" if None in s else
          f"  vs {v:18s} {s[0]['wins'] + s[1]['wins']}/40 ({(s[0]['mean_score'] + s[1]['mean_score']) / 2:.1f}; {s[0]['mean_score']:.0f}/{s[1]['mean_score']:.0f})")
EOF
grep -h '"validation_agreement"' -A 40 runs/$NAME.training.json | grep -E '"(all|footman|archer|cavalry|teacher_driven|learner_driven)"' | head -20 || true
cat runs/${NAME}_eval/behavior.json
