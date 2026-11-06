#!/bin/bash
# Headless runner for an 8-GPU Hyperstack VM (passed as cloud-init user_data).
# The launcher prepends:  GH_TOKEN=...  BRANCH=...  BUDGET_S=...
# Results/logs are pushed back to $BRANCH under research/dcbench/results/gpu/.
exec > /var/log/dcbench.log 2>&1
set -x
export HOME=/root DEBIAN_FRONTEND=noninteractive
REPO=https://x-access-token:${GH_TOKEN}@github.com/andrewliu2001/hnet.git
WORK=/ephemeral; [ -d $WORK ] || WORK=/root
cd $WORK

apt-get update -qq && apt-get install -y -qq git python3-pip >/dev/null
pip install -q --break-system-packages torch numpy pyarrow || pip install -q torch numpy pyarrow

git clone -q -b "$BRANCH" "$REPO" hnet
cd hnet/research/dcbench
git config user.name "dcbench-vm"; git config user.email "dcbench-vm@users.noreply.github.com"
R=results/gpu; mkdir -p $R/logs

push() {  # commit whatever results exist and push (rebasing on concurrent pushes)
  cp /var/log/dcbench.log $R/logs/vm.log 2>/dev/null
  for f in $WORK/logs/*.log; do [ -f "$f" ] && tail -n 300 "$f" > $R/logs/$(basename "$f"); done
  git add -A $R && git commit -qm "dcbench GPU: $1" || true
  for i in 1 2 3 4 5; do git pull -q --rebase origin "$BRANCH" && git push -q origin HEAD:"$BRANCH" && break; sleep $((i*5)); done
}
nvidia-smi > $R/logs/nvidia-smi.txt; python3 -c "import torch;print(torch.__version__, torch.cuda.device_count())" >> $R/logs/nvidia-smi.txt
echo "setup" > $R/STATUS; push "VM up"

python3 prepare_fineweb.py --shards 2 --out $WORK/data > $R/logs/prepare.log 2>&1
D="--train $WORK/data/train.bin --val $WORK/data/val.bin"

# speed probe (IF hits its target ratio from the start; H-Net drifts, hence the margin)
python3 train_gpu.py --chunker if_last $D --probe 120 > $R/logs/probe.log 2>&1
SPS=$(grep PROBE $R/logs/probe.log | awk '{print $3}')
if [ -z "$SPS" ]; then echo "probe failed" > $R/STATUS; push "probe FAILED"; exit 1; fi
STEPS=$(python3 -c "print(int($BUDGET_S / ($SPS * 1.25) // 1000 * 1000))")
echo "probe ${SPS}s/step -> steps $STEPS" >> $R/logs/prepare.log
echo "training steps=$STEPS" > $R/STATUS; push "probe ${SPS}s/step, steps $STEPS"

mkdir -p $WORK/logs
PIDS=()
RUNS=("fixed 1" "fixed 2" "hnet 1" "hnet 2" "hnet 3" "if_last 1" "if_last 2" "if_last 3")
for i in "${!RUNS[@]}"; do
  set -- ${RUNS[$i]}
  CUDA_VISIBLE_DEVICES=$i nohup timeout $((BUDGET_S * 2)) python3 train_gpu.py --chunker $1 --seed $2 $D \
    --steps $STEPS --out $R > $WORK/logs/$1_s$2.log 2>&1 &
  PIDS+=($!)
done
( while true; do sleep 900; push "progress"; done ) & PUSHER=$!
wait "${PIDS[@]}"
kill $PUSHER
echo "done" > $R/STATUS; push "all runs finished"
