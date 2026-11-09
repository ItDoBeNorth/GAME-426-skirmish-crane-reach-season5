#!/usr/bin/env bash
# One encoder round (S5 Phase D): collect with the current encoder, train, freeze, evaluate.
# Usage, from the repo folder:  bash tools/feature_round.sh NAME [EXTRA_OPPONENT_FOLDER ...]
#   1. collect teacher demonstrations vs Banner and self-play, seeds 0-19 (8 processes)
#   2. train on them (validation seeds 8 9 18 19) to runs/NAME.pt
#   3. freeze the learner as rivals/l_NAME (agent.py, encoder.py, imitation_model.py, model)
#   4. agreement on the validation seeds + games on seeds 600-619 vs Banner, the teacher
#      (rivals/s5_t5b), the shipped Banner learner and PREVIOUS_LEARNER_FOLDER, both seats
# Results: runs/NAME/ (data, logs) and runs/NAME_eval/ (agreement and game summaries).
set -euo pipefail
NAME=$1
PREVIOUS="${@:2}"
PY=.venv/Scripts/python.exe
mkdir -p runs/$NAME runs/${NAME}_eval

for start in 0 5 10 15; do
  $PY -m imitation.experiment collect --episodes 5 --start $start --opponent rivals/banner --output runs/$NAME/banner_$start.jsonl > runs/$NAME/banner_$start.log 2>&1 &
  $PY -m imitation.experiment collect --episodes 5 --start $start --opponent teacher --output runs/$NAME/self_$start.jsonl > runs/$NAME/self_$start.log 2>&1 &
done
wait
tail -qn1 runs/$NAME/*.log

$PY -m imitation.train runs/$NAME/banner_*.jsonl runs/$NAME/self_*.jsonl --validation-seeds 8 9 18 19 --output runs/$NAME.pt > runs/$NAME/train.log 2>&1
grep -v 'Joint policy' runs/$NAME/train.log | grep -v '^\s*$'

FOLDER=rivals/l_$NAME
mkdir -p $FOLDER
cp agent.py encoder.py imitation_model.py manifest.json $FOLDER/
for helper in routes.py enemy_memory.py; do [ -f $helper ] && cp $helper $FOLDER/; done
cp runs/$NAME.pt $FOLDER/imitation-policy.pt

$PY tools/analyze_disagreements.py runs/$NAME.pt runs/$NAME/banner_*.jsonl runs/$NAME/self_*.jsonl --validation-seeds 8 9 18 19 > runs/${NAME}_eval/agreement.txt 2>&1 &
OPPONENTS="rivals/banner rivals/s5_t5b rivals/banner_learner $PREVIOUS"
for seat in 0 1; do
  for vs in $OPPONENTS; do
    $PY tools/baseline.py $FOLDER $vs $seat 600 20 > runs/${NAME}_eval/$(basename $vs)_$seat.txt 2>&1 &
  done
done
wait
cat runs/${NAME}_eval/agreement.txt
$PY - "$NAME" $OPPONENTS <<'EOF'
import json, os, sys
name, opponents = sys.argv[1], sys.argv[2:]
def summ(p):
    if not os.path.exists(p): return None
    for line in open(p):
        if line.startswith("SUMMARY"): return json.loads(line[8:])
print(f"{name} on seeds 600-619, wins/40 (mean; red/blue; mean act ms)")
for vs in opponents:
    v = os.path.basename(vs)
    s = [summ(f"runs/{name}_eval/{v}_{seat}.txt") for seat in (0, 1)]
    if None in s:
        print(f"  vs {v}: ERROR"); continue
    print(f"  vs {v:18s} {s[0]['wins'] + s[1]['wins']}/40 ({(s[0]['mean_score'] + s[1]['mean_score']) / 2:.1f}; "
          f"{s[0]['mean_score']:.0f}/{s[1]['mean_score']:.0f}; {(s[0]['mean_act_ms'] + s[1]['mean_act_ms']) / 2:.1f} ms)")
EOF
