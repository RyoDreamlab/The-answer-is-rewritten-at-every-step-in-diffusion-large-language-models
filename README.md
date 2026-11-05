# The answer letter is rewritten at the unmask step

These scripts change the letter written into the answer slot at the step that slot is unmasked. They are for diffusion language models. The models are LaViDa, Dream-VL, LLaDA-V, and MMaDA. The tasks are ScienceQA, M3CoT, MMBench, MMStar, POPE, MME, and V*.

The code is in `experiments/unmask_order/`. Prior subtraction changes letter logits by a content-free prior while the slot is still masked. Hysteresis, bump, and retune replace the unmask-step token with a token that same slot already wrote. The agree gate is an offline replay over saved write histories.

Run the scripts from `/mnt/pfs/peng_gao/visual-hysteresis` on that machine. One cell from `experiments/unmask_order/queue_release_20261026.sh`:

```bash
.venv/bin/python3 experiments/unmask_order/run_release_20261026.py e \
  --backend lavida --dataset pope --mode gated --alpha-rule spread \
  --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 \
  --output outputs/release_margin_20261013e/lavida_pope_gated.jsonl \
  --progress outputs/release_margin_20261013e/lavida_pope.progress.json
```
