#!/bin/bash
# Poll the ntfy topic the VM reports to; unpack each results tarball into the repo.
# usage: poll_ntfy.sh TOPIC VM_ID
T=$1; VM=$2; SUB=${3:-gpu}; DEST=/home/user/hnet/research/dcbench; SEEN=/tmp/ntfy_seen_$T; touch $SEEN
while true; do
  st=$(curl -sS -m 30 -H "api_key: $HYPERSTACK_API_KEY" https://infrahub-api.nexgencloud.com/v1/core/virtual-machines/$VM \
       | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('instance',{}).get('status','?'))" 2>/dev/null)
  curl -sS -m 30 "https://ntfy.sh/$T/json?poll=1&since=all" 2>/dev/null | python3 -c "
import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get('event')=='message' and m.get('attachment'): print(m['id'], m['attachment']['url'], m.get('title','').replace(' ','_'))
" | while read id url title; do
    grep -q "^$id$" $SEEN && continue
    echo $id >> $SEEN
    curl -sS -m 120 -o /tmp/r_$T.tgz "$url" && tar xzf /tmp/r_$T.tgz -C $DEST 2>/dev/null
    evals=$(grep -h "EVAL" $DEST/results/$SUB/logs/*_s*.log 2>/dev/null | tail -n 8 | sed -E "s/'word_prec'.*'step'/step/; s/, 'time'.*//")
    last=$(for f in $DEST/results/$SUB/logs/*_s*.log; do [ -f $f ] && grep " step " $f | tail -n1; done | cut -c1-90)
    echo "[$(date +%H:%M)] VM=$st msg=$title STATUS=$(cat $DEST/results/$SUB/STATUS 2>/dev/null)"
    [ -n "$last" ] && echo "$last"
    [ -n "$evals" ] && echo "$evals"
    grep -h -iE "Traceback|Error|FAILED|out of memory" $DEST/results/$SUB/logs/*.log 2>/dev/null | tail -n 3
  done
  case "$st" in ERROR|DELETED|SHUTOFF|"") echo "[$(date +%H:%M)] VM status: '$st'";; esac
  sleep 120
done
