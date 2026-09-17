#!/usr/bin/env bash
# RTLS 전이 실험: (1) ETH/UCY 학습 가중치로 미세조정 (2) RTLS 에서 처음부터 (빠른 설정) (3) 세 모델을 같은 테스트로 평가
#   scripts/run_rtls_transfer.sh                # 기본: init eth seed0, 학습 장면 60,000개, every=10 평가
set -euo pipefail
cd "$(dirname "$0")/.."
export FORESIGHT_ROOT="$PWD" MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 OMP_WAIT_POLICY=PASSIVE
INIT="${INIT:-results/checkpoints/eth/seed0/best.pth}"; LIMIT="${LIMIT:-60000}"; EVERY="${EVERY:-10}"
DATA="${DATA:-data/rtls/full/processed}"
mkdir -p results/logs
echo "[$(date +%H:%M:%S)] finetune from $INIT"
foresight train dataset=rtls dataset.data_dir="$DATA" dataset.limit_scenes=$LIMIT train=finetune train.mode=bucket train.bucket_batch=32 \
  init_from="$INIT" run_name=rtls-finetune-eth mlflow.experiment=social-stgcnn-rtls threads=1 eval.seeds=[0] > results/logs/rtls-finetune.log 2>&1
echo "[$(date +%H:%M:%S)] scratch (fast)"
foresight train dataset=rtls dataset.data_dir="$DATA" dataset.limit_scenes=$LIMIT train=fast run_name=rtls-scratch-fast \
  mlflow.experiment=social-stgcnn-rtls threads=1 eval.seeds=[0] > results/logs/rtls-scratch.log 2>&1
echo "[$(date +%H:%M:%S)] evaluate-rtls"
foresight evaluate-rtls --data-dir "$DATA" --every "$EVERY" --seeds 0,1,2 \
  --ckpts "$INIT,results/checkpoints/rtls-finetune-eth/best.pth,results/checkpoints/rtls-scratch-fast/best.pth" \
  --names "zero-shot-eth,finetuned-eth,scratch-rtls" > results/logs/evaluate-rtls.log 2>&1
echo "[$(date +%H:%M:%S)] done"
