#!/bin/bash
# Headless runner for the OFFICIAL H-Net collapse check (train_official.py).
# The launcher prepends:  TOPIC=...  BRANCH=...  STEPS=...  RESULTS_DIR=...  RUNS_SPEC="variant:seed ..."  [SMOKE=1]
# No credentials live on the VM (public repo; results go to an unguessable ntfy.sh topic).
exec > /var/log/dcbench.log 2>&1
set -x
export HOME=/root DEBIAN_FRONTEND=noninteractive
WORK=/ephemeral; [ -d $WORK ] || WORK=/root
cd $WORK

apt-get update -qq && apt-get install -y -qq git python3-pip >/dev/null
PIP="pip install -q --break-system-packages"
$PIP torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$PIP numpy pyarrow einops optree "triton==3.2.0"
GH=https://github.com
$PIP $GH/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
$PIP $GH/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
$PIP $GH/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

git clone -q -b "$BRANCH" https://github.com/andrewliu2001/hnet.git hnet
cd hnet/research/dcbench
R=${RESULTS_DIR:-results/official}; mkdir -p $R/logs $WORK/logs

push() {
  cp /var/log/dcbench.log $R/logs/vm.log 2>/dev/null
  for f in $WORK/logs/*.log; do [ -f "$f" ] && tail -n 300 "$f" > $R/logs/$(basename "$f"); done
  tar czf /tmp/results.tgz $R
  for i in 1 2 3; do curl -sS -m 60 -T /tmp/results.tgz -H "Filename: results.tgz" -H "Title: $1" \
    "https://ntfy.sh/$TOPIC" > /dev/null && break; sleep 10; done
}
python3 -c "import torch, mamba_ssm, flash_attn, causal_conv1d; print(torch.__version__, torch.cuda.device_count(), mamba_ssm.__version__, flash_attn.__version__)" > $R/logs/env.txt 2>&1
nvidia-smi >> $R/logs/env.txt
echo "setup" > $R/STATUS; push "VM up"

python3 prepare_fineweb.py --shards 2 --out $WORK/data > $R/logs/prepare.log 2>&1
D="--train $WORK/data/train.bin --val $WORK/data/val.bin"

if [ -n "$SMOKE" ]; then
  for v in hnet fixed; do
    python3 train_official.py --variant $v $D --probe 120 > $WORK/logs/probe_$v.log 2>&1
    python3 train_official.py --variant $v $D --steps 400 --warmup 100 --eval_every 400 --eval_batches 20 \
      --out $R > $WORK/logs/smoke_$v.log 2>&1
  done
  echo "done" > $R/STATUS; push "smoke finished"; exit 0
fi

echo "training steps=$STEPS" > $R/STATUS; push "steps $STEPS"
PIDS=(); i=0
for r in $RUNS_SPEC; do
  v=${r%%:*}; s=${r##*:}
  CUDA_VISIBLE_DEVICES=$i nohup python3 train_official.py --variant $v --seed $s $D --steps $STEPS --out $R \
    > $WORK/logs/official_${v}_s$s.log 2>&1 &
  PIDS+=($!); i=$((i+1))
done
( while true; do sleep 900; push "progress"; done ) & PUSHER=$!
wait "${PIDS[@]}"
kill $PUSHER
echo "done" > $R/STATUS; push "all runs finished"
