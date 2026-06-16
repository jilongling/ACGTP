# ACGTP 核心代码与 50% 视觉 Token 剪枝时延审计（2026-05-31）

## 结论先行

本轮检查的结论比较明确：**当前 50% 视觉 token 剪枝确实生效，进入 LLM 内部的视觉 token 从 256 降到 128；branch attribution 也基本真实记录；但它没有带来稳定 latency 提升。** 主要原因不是 selector 没有剪掉 token，而是当前 base OpenVLA 的系统路径让这部分计算下降很难转化为端到端速度：自回归 decode 占大头、剪枝发生在 LLM 第 2 层之后、vision encoder/projector 没有被剪短、hook 约 3.9ms 的额外开销吃掉收益，并且本次 10 task x 1 episode 中真实 LLM prefill/total 反而没有下降。

因此当前能稳妥宣称的是：**ACGTP functional quota 已验证 token compression / analytic FLOPs / profiler FLOPs 层面的机制减算；尚未验证 base OpenVLA 上的稳定 wall-clock/control-frequency 加速。**

## 本轮检查范围

代码文件：

- `/infini-data/openvla/pruning/internal_pruning.py`
- `/infini-data/openvla/pruning/hook.py`
- `/infini-data/openvla/pruning/selector.py`
- `/infini-data/openvla/pruning/metrics.py`
- `/infini-data/openvla/utils/metrics_logger.py`
- `/infini-data/openvla/scripts/eval_openvla_baseline.py`
- `/infini-data/openvla/scripts/build_benchmark_metrics.py`
- `/infini-data/openvla/scripts/rollup_pruning_results.py`
- `/infini-data/openvla/scripts/run_functional_quota_ablation.py`

结果文件：

- `/infini-data/openvla/outputs/function_quota_10task1ep_instrumented_20260531_1708/benchmark_report.md`
- `/infini-data/openvla/outputs/function_quota_10task1ep_instrumented_20260531_1708/benchmark_comparison.csv`
- `/infini-data/openvla/outputs/function_quota_10task1ep_instrumented_20260531_1708/*/step_metrics.csv`
- `/infini-data/openvla/outputs/profiler_flops_smoke_20260531/benchmark_metrics.json`

## 当前策略梳理

| 策略 | 作用 | 当前定位 |
|---|---|---|
| `baseline_none` | 不剪枝，256 个视觉 token 全量进入 LLM | 质量、时延、FLOPs、显存、Control Freq. 基线 |
| `legacy_geo_guarded_quota_050` | 旧版 geo guarded internal pruning，keep_ratio=0.5，近似为 `P_hard_geo ∪ P_semantic_proxy ∪ P_fill` | 旧主线/对照。最新 10x1 成功率 7/10，出现质量回退 |
| `functional_quota_static_050` | 当前主策略，keep_ratio=0.5，`P_hard_geo ∪ P_layout ∪ P_contact ∪ P_motion ∪ P_sem ∪ P_act ∪ P_fill` | 值得继续作为主线优化的 structured allocation 策略；质量较稳，但本轮没有时延收益 |
| `functional_no_layout_050` | 去掉 layout quota | 验证 scene-layout 分支重要性 |
| `functional_no_contact_050` | 去掉 contact quota | 验证 contact/interaction 分支重要性 |
| `functional_no_motion_050` | 去掉 motion quota | 验证 motion-corridor 分支重要性。当前质量不差，说明 motion 可能需要 adaptive/optional 化 |

补充：`functional_no_semantic_050` 与 `functional_no_fill_050` 已在 runner 中定义，但不在最新 6 策略正式表里。

## 最新 10 task x 1 episode 结果摘要

来源：`/infini-data/openvla/outputs/function_quota_10task1ep_instrumented_20260531_1708/benchmark_comparison.csv`。

| 策略 | 成功 | 视觉 token | LLM total | prefill | decode | hook | model speed | wall speed | Control Freq. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_none | 8/10 | 256 | 212.40ms | 44.18ms | 168.22ms | N/A | 1.000x | 1.000x | 3.66Hz |
| functional_quota_static_050 | 9/10 | 128 | 219.39ms | 46.58ms | 172.81ms | 3.91ms | 0.956x | 0.952x | 3.49Hz |
| legacy_geo_guarded_quota_050 | 7/10 | 128 | 212.16ms | 46.06ms | 166.10ms | 3.90ms | 0.987x | 0.980x | 3.59Hz |
| functional_no_layout_050 | 8/10 | 128 | 216.73ms | 46.24ms | 170.49ms | 3.96ms | 0.966x | 0.960x | 3.52Hz |
| functional_no_contact_050 | 8/10 | 128 | 217.36ms | 46.34ms | 171.02ms | 3.92ms | 0.964x | 0.957x | 3.51Hz |
| functional_no_motion_050 | 9/10 | 128 | 214.66ms | 46.00ms | 168.66ms | 3.88ms | 0.976x | 0.969x | 3.55Hz |

关键读法：

- `speedup > 1` 才是加速；当前所有剪枝策略的 `model_speedup` 和 `wall_speedup` 都小于 1，是变慢。
- `functional_quota_static_050` 保持/略高于 baseline 成功数，但 LLM total 比 baseline 慢约 6.99ms，prefill 也慢约 2.40ms。
- Control Freq. 已实现，按 `1000 / mean_step_wall_time_ms_step_records` 计算，包含 model forward、hook、action postprocess 和 `env.step`，不含 settling/no-op steps。

## 剪枝过程是否真的生效

结论：**生效。**

证据：

- `step_metrics.csv` 中 `internal_original_visual_tokens=256`，`internal_kept_visual_tokens=128`，每步都有真实值。
- functional 主策略第一批样本显示 `internal_original_seq_length=291`，`internal_kept_seq_length=163`。视觉 token 保留率是 50%，但由于文本 token 永远保留，实际 LLM 序列保留率约为 55.5%。
- `branch_accounting_valid=True`，functional 主策略的 `unique_sum=128`，说明最终 keep token 能被 branch ownership 解释。
- `internal_pruned_geo_critical_count=0`，几何硬保护没有被普通剪枝删掉。

所以当前不是“没有剪掉 token”，也不是“benchmark 假装剪枝”。真正的问题是：**剪掉的这部分视觉 token 没有转化成稳定时延下降。**

## 核心代码路径审计

### 1. 实验 runner 与策略开关

`/infini-data/openvla/scripts/run_functional_quota_ablation.py:22-31` 定义了 internal backend 的共同配置：

- `--pruning_strategy robot_geo_acgtp_v2`
- `--keep_ratio 0.50`
- `--acgtp_compression_backend internal`
- `--acgtp_internal_pruning_enabled true`
- `--acgtp_internal_selection_mode geo_guarded`
- `--acgtp_dynamic_enabled false`

`/infini-data/openvla/scripts/run_functional_quota_ablation.py:34-67` 定义了 baseline、functional 主策略、no-layout/no-contact/no-motion/no-semantic/no-fill 和 legacy 对照。这里没有发现策略名映射错乱。

### 2. OpenVLA 视觉 token 位置仍是硬编码

`/infini-data/openvla/pruning/hook.py:1484-1488`：

- `image_token_start_index=1`
- `image_token_length=256`

这对当前 base OpenVLA-7B 是可用的，但仍是 OpenVLA 专用硬编码。它不是本轮 latency 失败的直接原因，但会影响以后推广到 OpenVLA-OFT / pi0。

### 3. internal 模式不会剪短 projector 输出

`/infini-data/openvla/pruning/hook.py:4461-4472` 显示 internal mode 下：

- 只准备 internal pruning plan；
- `metrics.timing.gather_ms = 0.0`；
- `pruned = visual_tokens`；
- projector 输出本身不 gather、不变短。

这点非常关键：**vision encoder 和 projector 不会因为 internal pruning 变快。** 剪枝收益只能发生在 LLM 的剪枝层之后。

### 4. 真正剪枝发生在 LLM 第 2 层之后

`/infini-data/openvla/pruning/internal_pruning.py:1103-1155` 是核心路径：

- 每层循环到 `layer_idx == plan.prune_layer` 时触发选择；
- 默认 `prune_layer=2`；
- 如果没有 materialized attention，就用 `_qk_text_to_visual_attention` 代理；
- 得到 `visual_keep` 后，通过 `hidden_states.index_select(1, keep_indices)` 真正缩短 hidden states；
- 同步更新 `position_ids`、`cache_position` 和 `causal_mask`。

因此剪枝不是假的，但它只节省第 3 层及之后的 LLM 计算；第 0/1/2 层仍然跑完整视觉序列。

### 5. functional quota 选择逻辑符合“配额并集”主线

`/infini-data/openvla/pruning/internal_pruning.py:715-958` 体现了当前主策略：

- 显式 `geo_protect_mask` 先加入，且超预算时提升 `target_k`；
- functional quota 按 layout/contact/motion/sem/hist/fill 分配；
- semantic/history attention 如果启用 geometry alignment，只在 `geo_score > 0` 的区域里选；
- 最终优先级是 `protect_keep_list -> geo_keep_list -> other_keep_list`；
- 输出 selected counts、unique counts、overlap ratio、accounting valid。

这与 ACGTP 的“不是全局 top-k，而是功能分支预算并集”方向一致。

### 6. 当前语义 attention 仍是 QK proxy，不是真实 materialized attention

`/infini-data/openvla/pruning/internal_pruning.py:1113-1136` 支持 `try_output_attentions`，但 `/infini-data/openvla/pruning/hook.py:843-876` 的 `quota_config` 没有传入 `try_output_attentions`。

最新日志中：

- `internal_attention_source=llm_qk_text_to_vision`
- `internal_historical_action_attention_available=False`

因此当前 `P_sem` 是 QK 投影代理，`P_act` 还没有真实历史 action attention。这个问题主要影响策略语义质量与论文表述，**不是 50% token 剪枝没有加速的主因**。

### 7. LLM prefill/decode 拆分是真实 instrumentation，但会同步 CUDA

`/infini-data/openvla/scripts/eval_openvla_baseline.py:2929-3018` 注册 language_model forward hook，通过 `seq_len` 和 cache/past 状态把调用分成 prefill 与 decode。每次 pre/post hook 都会 `torch.cuda.synchronize()`。

结论：

- prefill/decode 字段不是伪造值，当前日志可用；
- 但这属于诊断模式，绝对时延会包含同步开销；正式 latency 表建议同时跑一版低 instrumentation 的 wall-clock benchmark。

### 8. speedup 公式方向正确

`/infini-data/openvla/scripts/build_benchmark_metrics.py:640-705` 使用：

- `speedup = baseline_latency / method_latency`
- `latency_reduction = baseline_latency - method_latency`

因此当前报告没有把 `0.95x` 错写成加速；`wall_clock_not_accelerated` warning 是正确的。

## 发现的逻辑漏洞与诊断缺口

### A. 影响“策略语义”的问题

1. **P_sem 不是论文级真实 attention**
   - 当前来自 `llm_qk_text_to_vision`，不是 FlashAttention 下 materialized attention。
   - 只能写成 QK proxy / attention proxy，不能宣称 true decoder attention。

2. **P_act 尚不可用**
   - `internal_historical_action_attention_available=False`。
   - 当前策略名里有 action/hist 分支，但实际没有动作 decode attention 贡献。

3. **OpenVLA token 位置硬编码**
   - `image_token_start_index=1`、`image_token_length=256`。
   - 当前验证没问题，但泛化到 OFT/pi0 前必须抽象。

### B. 影响“诊断完整性”的问题

1. **显式 geo hard-protect 两个字段没有进 step CSV**
   - internal info 已生成 `internal_geo_explicit_protected_count` 和 `internal_budget_raised_for_geo_protection`；
   - 但 `eval_openvla_baseline.py` 的 step allow-list 里没有这两个字段，最新 `step_metrics.csv` header 也没有。
   - 这不影响剪枝执行，但影响硬保护审计完整性。

2. **strict paired 仍缺失**
   - 最新 benchmark 中 `strict_paired_n=0`，说明这批正式结果还不能称为严格 seed/init-state paired。
   - 只能作为 task-trial paired / medium noisy evidence。

3. **summary.json 太臃肿，不适合作论文表格**
   - 当前应优先使用 `benchmark_report.md` 和 `benchmark_comparison.csv`。
   - `summary.json` 适合保留原始全量聚合，不适合人工解读。

### C. 影响“时延解释”的问题

1. **projector_time_ms 在剪枝策略下实际包含 hook 成本**
   - baseline projector 约 0.36ms；剪枝策略 projector 约 4.1ms；hook_total 约 3.9ms。
   - 说明 projector timing 读数里基本叠入了 projector hook 的执行。
   - 分析时不能再把 projector delta 和 hook_total 当两个完全独立开销重复相加。

2. **冷启动尖峰存在，但不是主因**
   - functional 主策略第一条记录 prefill 可到 317ms、projector 可到 23ms。
   - 但剔除每个 episode 第一条记录后，functional prefill 仍约 46.35ms，高于 baseline 的 44.09ms；hook 仍约 3.90ms。
   - 所以“不加速”不是单纯 first-step 污染。

3. **同样 50% retention 下 LLM total 差异较大**
   - 各 50% 策略 LLM total 从约 212ms 到 219ms。
   - 这说明 10x1 的任务轨迹、decode 长度、环境状态与 GPU 噪声仍会影响均值；正式表需要 strict paired + 多 episode。

## 为什么剪掉 50% 视觉 token 后没有本质加速

### 原因 1：base OpenVLA 的自回归 decode floor 太厚

baseline LLM total 约 212.40ms，其中：

- prefill 约 44.18ms；
- decode 约 168.22ms。

也就是说 LLM 时间里 decode 占约 79%。ACGTP 当前视觉 token 剪枝主要影响 prefill 和后续上下文长度，对 base OpenVLA 的自回归 action token decode 地板影响有限。就算 prefill 真的省 20%，反映到 LLM total 也只是几个 ms 级别，进一步到 wall-clock 更小。

### 原因 2：剪枝发生在第 2 层之后，不是从第 0 层开始省

当前默认 `prune_layer=2`，所以 LLM 的第 0/1/2 层仍跑完整序列。剪枝只影响后面的层。analytic FLOPs 已经按这一点估算，但真实 kernel latency 不一定线性跟随 FLOPs。

### 原因 3：实际序列保留率不是 50%，而是约 55.5%

视觉 token 从 256 到 128 是 50%，但文本 token 永远保留。以 functional 主策略样本为例：

- 原始 seq length 约 291；
- 剪后 seq length 约 163；
- 内部序列保留率约 55.5%。

因此不要把“剪掉 50% 视觉 token”直接等同于“LLM 总计算减少 50%”。

### 原因 4：vision encoder / projector 不受 internal pruning 加速

internal mode 下 projector 输出不变短；剪枝计划在 projector hook 里准备，真正裁剪发生在 LLM 内部。结果是：

- vision encoder 仍约 19ms；
- projector baseline 约 0.36ms，但剪枝策略下变成约 4.1ms，主要来自 hook；
- 这部分不会因为 LLM token 减少而下降。

### 原因 5：hook overhead 约 3.9ms，收益窗口太小

functional 主策略：

- hook_total 约 3.91ms；
- selection 约 0.96ms；
- 本轮 LLM saving 是负数：`212.40 - 219.39 = -6.99ms`；
- net after hook 约 `-10.90ms`。

即使未来 prefill 真的省出 3-5ms，当前 hook 也足以吃掉大部分收益。hook 优化到 `<1ms` 前，很难在 base OpenVLA 上看到稳定 wall speedup。

### 原因 6：真实 measured prefill 没有下降

本轮不是“prefill 下降但 wall 没下降”，而是：

- baseline prefill 44.18ms；
- functional 主策略 prefill 46.58ms。

即使剔除每个 episode 第一条记录，functional prefill 仍高于 baseline。说明当前 GPU kernel、同步、轨迹差异、剪枝后 attention/cache 形状等因素，使 token 减少没有在 measured prefill latency 上兑现。

### 原因 7：论文中的加速机制通常比当前路径更“系统级”

与 VLA-Cache、DepthCache、VLA-Pruner、VLA-ADP、VLA-IAP 这类工作相比，当前 ACGTP 的差异是：

- 很多论文减少的是重复帧/静态 token 的 KV 或特征计算，或者在更早位置剪枝；当前 internal pruning 不省 vision/projector，也不复用 cache。
- 一些论文报告 FLOPs/token compression/model-only latency；当前如果看 analytic/profiler FLOPs，也确实下降，但 wall-clock 没穿透。
- 一些方法可能配合更低开销实现、batch/engine/kernel 优化或非 base action head；当前 base OpenVLA 自回归 decode 地板很强。
- 这些论文通常同时报告 success、latency/speedup、memory/FLOPs/token compression、Control Freq.；当前项目的指标已经基本对齐，但结论必须按真实数值写，不能把 FLOPs reduction 写成 wall speedup。

## FLOPs 与时延的关系

当前有两层证据：

1. **analytic LLM FLOPs**
   - 来自 logged seq length + LLM config；
   - 对 functional 主策略给出约 `1.66x` LLM-block FLOPs speedup；
   - 这是理论/解析估计，不是硬件 profiler。

2. **torch.profiler FLOPs smoke**
   - `/infini-data/openvla/outputs/profiler_flops_smoke_20260531/benchmark_metrics.json` 显示：
   - baseline profiler FLOPs per profiled step 约 `4.3306e12`；
   - functional 主策略约 `2.8042e12`；
   - profiler FLOPs speedup 约 `1.54x`，reduction 约 `35.25%`。

这说明计算量确实减少。注意：profiler run 会严重污染 latency，不能用它的 wall/model 时间做速度宣称。

## branch-level 机制是否真实记录

functional 主策略的 branch attribution 已经可用：

| 指标 | functional_quota_static_050 |
|---|---:|
| selected hard geo | 52 |
| selected layout | 69 |
| selected contact | 91 |
| selected motion | 66 |
| selected semantic | 18 |
| selected historical action | 0 |
| selected fill | 92 |
| unique hard geo | 52 |
| unique layout | 25 |
| unique contact | 17 |
| unique motion | 11 |
| unique semantic | 10 |
| unique historical action | 0 |
| unique fill | 12 |
| unique sum | 128 |
| overlap ratio | 66.8% |
| accounting valid | True |

读法：

- selected counts 是“某分支候选且最终幸存”的数量，分支之间可以重叠，所以加和会超过 128。
- unique counts 是最终 owner attribution，加和等于 128。
- 高 overlap 不是 bug，而是功能分支在同一 token 上有重合；但也说明当前 functional 分支之间存在较强冗余。

## 当前指标可靠性判断

| 指标 | 当前状态 | 是否可用于论文 |
|---|---|---|
| visual token 256 -> 128 | 真实日志 | 可用 |
| effective seq retention 约 55.5% | 真实日志/离线计算 | 可用 |
| LLM total | language_model forward hook | 可用，但注明 timing scope |
| prefill/decode split | forward hook 按 seq/cache 分类 | 可用，但属于诊断 instrumentation |
| hook_total/selection | hook 内真实计时 | 可用；selection 已包含在 hook_total 内 |
| model/wall speedup | baseline/method latency 比值 | 可用；当前是 slowdown |
| Control Freq. | `1000 / step wall` | 可用；当前低于 baseline |
| analytic FLOPs | seq length + model config 估算 | 可用作 analytic proxy，不是 exact hardware FLOPs |
| torch.profiler FLOPs | dedicated profiler smoke | 可用作 FLOPs 证据，不能用其 latency |
| memory | episode metrics | 可用；没有下降就如实报告 |
| cache hit/reuse | 不适用 | 不应报告为有效指标 |
| branch selected/unique/overlap | 真实日志 | 可用 |
| explicit geo protect count/budget raised | internal info 有，step CSV 缺 | 需补 allow-list 后重跑 |
| strict paired timing | 当前正式 run 缺 | 需重跑 |

## 能宣称什么，不能宣称什么

可以宣称：

- ACGTP internal pruning 在当前 OpenVLA-7B 路径里确实把 visual tokens 从 256 降到 128。
- functional quota 的 branch attribution 已真实记录，unique sum 与 kept token 对齐。
- analytic/profiler FLOPs 显示 LLM/model predict_action 计算量下降。
- functional 主策略在当前 10x1 小规模 run 中成功数不低于 baseline。
- 当前 benchmark 指标已基本覆盖 VLA 高效推理论文常用的 success、latency、speedup、Control Freq.、token compression、FLOPs、memory、branch diagnostic。

不能宣称：

- 不能说 functional quota 已实现稳定端到端加速。
- 不能说当前 base OpenVLA 上 Control Freq. 提升；实际从 3.66Hz 降到 3.49Hz。
- 不能说 prefill speedup 已证明；本轮 prefill 实测更慢。
- 不能说 P_sem/P_act 已经是完整真实 LLM attention；P_sem 仍是 QK proxy，P_act 当前不可用。
- 不能把 50% visual token pruning 写成 50% total FLOPs reduction。
- 不能把 profiler smoke 的 latency 当作正式 latency。

## 后续建议

### P0：把报告口径固定下来

- 对外表格以 `benchmark_comparison.csv` / `benchmark_report.md` 为准；`summary.json` 只保留原始聚合。
- 正式论文表里明确区分：token compression、analytic/profiler FLOPs、LLM total、prefill/decode、model latency、wall latency、Control Freq.。
- 任何 `speedup < 1` 都写成 slowdown。

### P1：补齐两个硬保护诊断字段

把下面两个 internal info 字段加入 step allow-list / metrics logger：

- `internal_geo_explicit_protected_count`
- `internal_budget_raised_for_geo_protection`

这不改变策略，只补审计闭环。

### P2：正式重跑 strict paired benchmark

建议用新 metadata 后重跑：

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python scripts/run_functional_quota_ablation.py \
  --output_root /infini-data/openvla/outputs/function_quota_strict_paired_10task1ep_YYYYMMDD_HHMMSS \
  --num_tasks 10 \
  --num_episodes 1 \
  --methods baseline_none,functional_quota_static_050,legacy_geo_guarded_quota_050,functional_no_layout_050,functional_no_contact_050,functional_no_motion_050

/infini-data/miniconda3/envs/openvla/bin/python scripts/build_benchmark_metrics.py \
  --input_dir /infini-data/openvla/outputs/function_quota_strict_paired_10task1ep_YYYYMMDD_HHMMSS \
  --baseline_strategy baseline_none
```

### P3：把 latency run 和 diagnostic run 分开

- latency run：尽量低 instrumentation，只保留必要 wall/model/LLM total/hook。
- diagnostic run：打开 prefill/decode split、branch attribution、profiler FLOPs。
- profiler FLOPs 必须独立 smoke，不和正式 latency 混用。

### P4：系统优化优先于继续堆策略

当前瓶颈不在“再加一个分支”，而在收益传导：

- hook_total 从约 3.9ms 压到 `<1ms`；
- 尽量避免 CPU/GPU/NumPy/Torch 往返；
- token grid、uv、固定 mask 预计算；
- layout 分支可做 temporal cache；
- 减少每步日志/可视化/诊断开销；
- 在不破坏质量的前提下评估更早 prune layer 或更低 keep ratio。

### P5：策略语义补课

- 若要对齐 VLA-Pruner 式语义/动作双目标，必须补真实 attention 或诚实标注 QK proxy。
- `try_output_attentions` 需要从 hook config 传入，但 FlashAttention 下可能退化/变慢，建议只用于 probe，不用于正式 latency。
- P_act 需要真实 decode attention/history buffer 才能称为 action attention。

## 总体判断

当前 ACGTP 的 functional quota 方向没有被否定：它已经从“单一几何 top-k”升级为“execution-function-aware structured token allocation”，并且 token/FLOPs/branch 机制是有证据的。真正的问题是：**在 base OpenVLA 的自回归 decode 地板和当前 hook 实现下，50% internal visual-token pruning 的计算收益还没有穿透到 prefill、model forward、wall-clock 和 Control Freq.。**

下一阶段应聚焦“系统级收益传导”：严格 paired 重跑、分离 latency/diagnostic、压低 hook、补硬保护日志、再评估 keep ratio / prune layer，而不是继续堆新的选择分支。