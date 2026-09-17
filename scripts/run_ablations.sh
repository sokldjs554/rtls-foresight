#!/usr/bin/env bash
# ablation: 공식 코드 동작(기본값) vs 논문 서술/대안 동작. 기본 분할 eth, 250 epoch 논문 설정 그대로.
#   scripts/run_ablations.sh            # 4개 ablation, 3 병렬
#   SPLIT=zara1 PAR=2 scripts/run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export FORESIGHT_ROOT="$PWD" MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
SPLIT="${SPLIT:-eth}"; PAR="${PAR:-3}"; SEED="${SEED:-0}"
mkdir -p results/logs
declare -A ABL=(
  [permute]="model.time_channel_swap=permute"            # 논문 그림대로 (C,T) 축 교환
  [poskernel]="model.graph.kernel=position"             # 논문 본문대로 위치 간 역거리 커널
  [bucket]="train.mode=bucket train.bucket_batch=32"    # 노드 수 같은 장면 묶어 배치 (BN 통계 배치 단위)
  [stableloss]="train.loss_exact=false"                 # 로그 영역 NLL (클램프 없음)
)
for name in "${!ABL[@]}"; do echo "$name ${ABL[$name]}"; done | xargs -P "$PAR" -L 1 bash -c '
  name=$0; shift; ov="$*"
  echo "[$(date +%H:%M:%S)] start ablation $name ($ov)"
  foresight train dataset='"$SPLIT"' train=paper seed='"$SEED"' threads=1 run_name=ablation-'"$SPLIT"'-$name mlflow.experiment=social-stgcnn-ablation $ov \
     > results/logs/ablation-'"$SPLIT"'-$name.log 2>&1 \
     && echo "[$(date +%H:%M:%S)] done  $name: $(grep -o "done:.*" results/logs/ablation-'"$SPLIT"'-$name.log)" \
     || echo "[$(date +%H:%M:%S)] FAIL  $name"
'
