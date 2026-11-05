# The answer letter is rewritten at every step

In these diffusion language models, the answer letter is rewritten at every step. The letter that counts is the one left in the answer slot when that slot is unmasked. The models are LaViDa, Dream-VL, LLaDA-V, and MMaDA. The tasks are ScienceQA, M3CoT, MMBench, MMStar, POPE, MME, and V*.

The code in `experiments/unmask_order/` leaves the unmask order unchanged and changes only that letter. It uses the letter scores at the unmask step and the letters that same slot has already written.

Run the scripts from `/mnt/pfs/peng_gao/visual-hysteresis` on that machine. One example from `experiments/unmask_order/queue_release_20261026.sh`:

```bash
.venv/bin/python3 experiments/unmask_order/run_release_20261026.py e \
  --backend lavida --dataset pope --mode gated --alpha-rule spread \
  --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 \
  --output outputs/release_margin_20261013e/lavida_pope_gated.jsonl \
  --progress outputs/release_margin_20261013e/lavida_pope.progress.json
```
