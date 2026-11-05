# 揭开答案槽时改写字母

扩散语言模型解码时，答案槽每一步都会被重写。槽被揭开的那一步，当时写在槽里的字母就是这道题留下的答案。本目录的脚本不改揭开顺序，只改这一步写进去的字母。

四个模型：LaViDa、Dream-VL、LLaDA-V、MMaDA。

七个任务：ScienceQA（`sciqa`）、M3CoT（`m3cot`）、MMBench（`mmbench`）、MMStar（`mmstar`）、POPE（`pope`）、MME（`mme`）、V*（`vstar`）。

脚本假定当前目录是 pro6000 上的这棵树：

`/mnt/pfs/peng_gao/visual-hysteresis`

解释器是该目录下的 `.venv/bin/python3`。权重、数据集、日志和 jsonl 轨迹不在这个 git 仓库里。脚本里用到的检查点路径是：

- `checkpoints/lavida-llada-v1.0-instruct`
- `checkpoints/MMaDA-8B-Base`
- `checkpoints/LLaDA-V`
- `checkpoints/Dream-VL-7B`
- `checkpoints/magvitv2`

## 模型 × 任务：用哪段命令

没有 28 个各自独立的启动文件。格子由共用解码器的 `--backend` 和 `--dataset` 决定。改进规则的 `run_*.py` 先装上规则，再把后面的参数原样交给对应解码器。

| 入口名 | 解码脚本 | 能跑的格子 |
|---|---|---|
| `e` | `calib_decode_multi_20261013e.py` | LaViDa、MMaDA 的 MMStar、POPE |
| `c` | `calib_decode_multi_20261013c.py` | LaViDa、MMaDA、LLaDA-V 的 ScienceQA、M3CoT、MMBench |
| `dream` | `calib_decode_dreamvl_20261013g.py` | Dream-VL 的 ScienceQA、M3CoT、MMBench、MMStar、POPE |
| `vstar` | `calib_decode_vstar_mme_20261022.py` | LaViDa、MMaDA、LLaDA-V 的 V*、MME |
| `dreamvm` | `calib_decode_vstar_mme_dreamvl_20261022.py` | Dream-VL 的 V*、MME |
| `lladav` | `calib_decode_lladav_e_20261022.py` | LLaDA-V 的 MMStar、POPE |

`run_release_20261026.py`、`run_priorhold_20261026.py`、`run_priortie_20261026.py`、`run_priortie10_20261026.py` 接受 `e|vstar|dream|dreamvm|lladav|c`。`run_release_20261026.py` 的用法字符串写的是 `{e|vstar|dream|c}`，文件里同样接 `dreamvm`。`run_hyst_20261029.py` 接受 `e|dream|dreamvm|lladav`。`run_bump_20261030.py` 接受 `e|c|dream|vstar`。`run_commitmargin_20261027.py` 接受 `dream|e`。`run_retune2_20261030.py` 只接受 `lavida_mme` 和 `dream_vstar`。

按模型看，每个任务的入口是：

| | ScienceQA | M3CoT | MMBench | MMStar | POPE | MME | V* |
|---|---|---|---|---|---|---|---|
| LaViDa | `c`，`--backend lavida --dataset sciqa` | `c`，`--backend lavida --dataset m3cot` | `c`，`--backend lavida --dataset mmbench` | `e`，`--backend lavida --dataset mmstar` | `e`，`--backend lavida --dataset pope` | `vstar`，`--backend lavida --dataset mme` | `vstar`，`--backend lavida --dataset vstar` |
| Dream-VL | `dream --dataset sciqa` | `dream --dataset m3cot` | `dream --dataset mmbench` | `dream --dataset mmstar` | `dream --dataset pope` | `dreamvm --dataset mme` | `dreamvm --dataset vstar` |
| LLaDA-V | `c`，`--backend lladav --dataset sciqa` | `c`，`--backend lladav --dataset m3cot` | `c`，`--backend lladav --dataset mmbench` | `lladav`，`--backend lladav --dataset mmstar` | `lladav`，`--backend lladav --dataset pope` | `vstar`，`--backend lladav --dataset mme` | `vstar`，`--backend lladav --dataset vstar` |
| MMaDA | `c`，`--backend mmada --dataset sciqa` | `c`，`--backend mmada --dataset m3cot` | `c`，`--backend mmada --dataset mmbench` | `e`，`--backend mmada --dataset mmstar` | `e`，`--backend mmada --dataset pope` | `vstar`，`--backend mmada --dataset mme` | `vstar`，`--backend mmada --dataset vstar` |

队列脚本把多格串在一起，用第一个参数选 GPU。例如 `queue_release_20261026.sh`、`queue_priorhold_20261026.sh`、`queue_priortie_20261026.sh`、`queue_priortie10_20261026.sh`、`queue_raise12_20261027.sh`、`queue_hyst9_20261029.sh`、`queue_bump1_20261030.sh` 都以 `0` 或 `1` 作为 GPU 号。`queue_sign_hard3_20261029.sh` 在脚本里把 GPU 写成 0。`follow_raise12_20261027.py` 在 GPU 空闲时接着跑 `queue_raise12_20261027.sh` 那一批格子。

## 内容无关先验相减

解码器的 `--mode gated` 在答案槽还盖着的时候，对选项字母的 logit 做 `-alpha * log p_cf`。`--alpha-rule spread` 时 `alpha_n = 1 - p_min/p_max`，再乘 `--alpha-scale`。先验文件由 `--prior-file` 传入。`--mode stock` 不改 logit。`--mode prior` 只计算内容无关先验，不解码。

在此之上，下面几个文件把「哪一步做减法」收窄，并由同名的 `run_*.py` 装进解码器：

- `release_margin_20261026.py`：只在选项 softmax 的前两名差距小于文件里的 `TAU`（0.10）时才减。`run_release_20261026.py` 负责装上它。队列是 `queue_release_20261026.sh`。
- `release_priorhold_20261026.py`：只在当前选项 argmax 仍是先验字母时才减。运行入口 `run_priorhold_20261026.py`。队列是 `queue_priorhold_20261026.sh` 和 `queue_priorhold_resume_20261026.sh`。
- `release_priortie_20261026.py`：argmax 仍是先验字母，且前两名差距小于 0.05。运行入口 `run_priortie_20261026.py`。队列是 `queue_priortie_20261026.sh`。
- `release_priortie10_20261026.py`：同样两条，差距阈值是 0.10。运行入口 `run_priortie10_20261026.py`。队列是 `queue_priortie10_20261026.sh`。
- `release_commitmargin_20261027.py`：更早的步保持原 logit，只改揭开那一步写入答案槽的 token。环境变量 `COMMIT_TAU`（默认 0.05）和 `COMMIT_MODE`（`margin_only` 或 `prior_and_margin`）。运行入口 `run_commitmargin_20261027.py`，只接 `dream` 和 `e`。`queue_raise12_20261027.sh` 里 Dream-VL M3CoT 和 MMaDA MMStar 走这个入口，其余格子直接调用解码器并带 `--alpha-scale`。

`queue_sign_hard3_20261029.sh` 对 LLaDA-V 的 MME、Dream-VL 的 MME 和 Dream-VL 的 V* 使用符号反过来的每步先验（`--alpha-scale=-0.1`），直接调用 `calib_decode_vstar_mme_20261022.py` 或 `calib_decode_vstar_mme_dreamvl_20261022.py`。

## hyst

`release_hyst_20261029.py` 不改揭开顺序。揭开时写入的 token 只能换成这个答案槽在更早一步已经写过的 token。规则放在环境变量 `HYST_RULE`：

- `back:K`：用揭开前第 K 步的字母
- `early_m:TAU`：选项差距低于 TAU 时，用前半段轨迹的多数字母
- `run:K`：末尾这个字母还没连续保持 K 步时，改回上一个不同的字母

`run_hyst_20261029.py` 装上这条规则。`check_hyst_20261029.py` 用库存轨迹核对一次 smoke 解码。队列是 `queue_hyst9_20261029.sh`。

## bump1

`release_bump_20261030.py` 同样只改揭开那一步，且新 token 必须是该槽写过的。`HYST_RULE` 为 `early_m:TAU`，或 `gate:R:S:T`（末尾字母保持不超过 R 步、轨迹切换不超过 S 次、选项差距不超过 T 时，改回上一个不同字母）。

`run_bump_20261030.py` 装上它。`check_bump_20261030.py` 做 smoke 核对。队列是 `queue_bump1_20261030.sh`。

## retune2

`release_retune2_20261030.py` 只改揭开答案槽时写入的 token，替换对象必须是该槽已经写过的 token。`RETUNE_RULE` 只能是 `lavida_mme` 或 `dream_vstar`，两条规则都看揭开步上选项 logit 的前两名差距。

`run_retune2_20261030.py` 装上它。`lavida_mme` 转到 V*/MME 解码器，`dream_vstar` 转到 Dream-VL 的 V*/MME 包装。`RETUNE_IDS` 可以限制 smoke 的题目 id。`check_retune2_20261030.py` 要么对已有 jsonl 做离线选择，要么核对一次新的 jsonl。

## release

这里的 release 就是上面的 margin release：减法仍是 `-alpha * log p_cf`，但只在字母分布还接近平局时使用，揭开步上已经分开的写入保持不动。文件是 `release_margin_20261026.py`，启动文件是 `run_release_20261026.py`，多格队列是 `queue_release_20261026.sh`。

## R6_agree_gate

`prereg_commit_rules_20261102.json` 写的是离线回放用的提交规则。`prereg_eval_20261102.py` 在 CPU 上读已经存好的库存轨迹，不启动 GPU 解码。`R6_agree_gate` 的做法是：若最后一次写入等于轨迹里的多数字母，就保留它；否则在这个槽真正写过的字母中，取「提交步 logit 减去 `alpha_n log p_cf`」的 argmax。

## 启动一格

在 `/mnt/pfs/peng_gao/visual-hysteresis` 下，`queue_release_20261026.sh` 里 LaViDa POPE 这一格的命令是：

```bash
.venv/bin/python3 experiments/unmask_order/run_release_20261026.py e \
  --backend lavida --dataset pope --mode gated --alpha-rule spread \
  --prior-file outputs/pope_20261013e/cf_pope_lavida.json --cuda 0 \
  --output outputs/release_margin_20261013e/lavida_pope_gated.jsonl \
  --progress outputs/release_margin_20261013e/lavida_pope.progress.json
```

换模型和任务时，把入口名（`e` / `c` / `dream` / `vstar` / `dreamvm` / `lladav`）以及 `--backend`、`--dataset` 换成上一节表格里的那一格。先验 json、输出路径和 `--cuda` 按那次运行自己的队列脚本填写。队列本身是：

```bash
bash experiments/unmask_order/queue_release_20261026.sh 0
```

`0` 是 GPU 号。`queue_bump1_20261030.sh` 在调用前设置 `HYST_RULE`，例如该文件里 LaViDa MMBench 使用 `gate:6:99:99`，然后：

```bash
.venv/bin/python3 experiments/unmask_order/run_bump_20261030.py c \
  --backend lavida --dataset mmbench --mode stock --cuda 0 \
  --output "$ROOT/outputs/bump1_20261030/mmbench_20261013c/lavida.jsonl" \
  --progress "$ROOT/outputs/bump1_20261030/mmbench_20261013c/lavida.progress.json"
```
