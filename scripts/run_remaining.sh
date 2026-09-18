#!/usr/bin/env bash
# 남은 실험 전부를 재시작 안전하게 (끝난 실행은 건너뛰고, 죽은 실행은 last.pth 에서 이어서) 돌린다.
#   nohup scripts/run_remaining.sh > results/logs-remaining.txt 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export FORESIGHT_ROOT="$PWD" PATH="$PWD/.venv/bin:$PATH" SKIP_DONE=1 DATA=data/rtls/full/processed MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 OMP_WAIT_POLICY=PASSIVE
mkdir -p results/logs
echo "[$(date +%H:%M:%S)] stage A: zara2 seed0 + RTLS 전이 + ablation(permute, poskernel)"
( SEEDS="0" SPLITS="zara2" PAR=1 scripts/train_all.sh > results/logs-train-all-seed0b.txt 2>&1 ) &
( scripts/run_rtls_transfer.sh > results/logs-rtls-transfer.txt 2>&1 ) &
( PAR=2 scripts/run_ablations.sh > results/logs-ablations.txt 2>&1 ) &
wait
echo "[$(date +%H:%M:%S)] stage B: seeds 1,2 × 5 분할 (PAR=4)"
SEEDS="1 2" PAR=4 scripts/train_all.sh > results/logs-train-all-seed12.txt 2>&1
echo "[$(date +%H:%M:%S)] stage C: 평가·수집·그림"
foresight evaluate --threads 2 > results/logs/evaluate-final.log 2>&1
python scripts/collect_results.py && python scripts/make_figures.py && python tools/check_readme_numbers.py && python scripts/sync_tables.py
echo "[$(date +%H:%M:%S)] all done"
