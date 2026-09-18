#!/usr/bin/env bash
# 논문 재현 학습: 5개 leave-one-out 분할을 병렬(기본 3 프로세스, 각 1 스레드)로 돌린다.
#   scripts/train_all.sh            # seed 0, paper 설정
#   SEEDS="0 1 2" PAR=4 scripts/train_all.sh
#   TRAIN=fast scripts/train_all.sh # ablation 설정
set -euo pipefail
cd "$(dirname "$0")/.."
export FORESIGHT_ROOT="$PWD" MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
SEEDS="${SEEDS:-0}"; PAR="${PAR:-3}"; TRAIN="${TRAIN:-paper}"; SPLITS="${SPLITS:-eth hotel univ zara1 zara2}"
EXTRA="${EXTRA:-}"
mkdir -p results/logs
jobs=()
for seed in $SEEDS; do for s in $SPLITS; do jobs+=("$s $seed"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c '
  s=$0; seed=$1
  if [ -f results/checkpoints/$s/seed$seed/metrics.json ] && [ "'"$TRAIN"'" = paper ]; then echo "[$(date +%H:%M:%S)] skip  $s seed=$seed (metrics.json exists)"; exit 0; fi
  echo "[$(date +%H:%M:%S)] start $s seed=$seed train='"$TRAIN"'"
  foresight train dataset=$s train='"$TRAIN"' seed=$seed threads=1 '"$EXTRA"' > results/logs/train-$s-'"$TRAIN"'-seed$seed.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] done  $s seed=$seed: $(grep -o "done:.*" results/logs/train-$s-'"$TRAIN"'-seed$seed.log)" \
    || echo "[$(date +%H:%M:%S)] FAIL  $s seed=$seed (see results/logs)"
'
