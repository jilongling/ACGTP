# OpenVLA ACGTP: Action-Constrained Geometric Token Protection

本项目基于 OpenVLA，在 LIBERO 推理阶段研究视觉 token 剪枝/保护策略。当前主线不是普通的视觉显著性剪枝，而是：

> 在减少 visual token 的同时，优先保留会约束机械臂未来动作的几何区域。

原始 OpenVLA README 已保留在 `oriREADME.md`。本文档只描述当前项目中的 ACGTP pipeline、代码结构、运行方式和实验验证流程。

## 当前状态

- 主策略：`robot_geo_acgtp_v2`
- 核心 baseline：`none`、`depth_edge_fast`、`depth_edge_fast_diverse`
- 当前推荐小规模对照 profile：`ACGTP dynamic-fast@0.75`
- 当前工程形态：projector 输出后做 visual token selection，然后将保留 token 交给 LLM
- 关键保护开关：`acgtp_position_preserve_enabled=True`
- 原始 LIBERO eval 会在 geometry strategy 下启用 depth observation，主要使用 `agentview_depth`

注意：项目中仍保留大量旧策略用于 ablation 和复现实验，但论文主线应以 `robot_geo_acgtp_v2` 为中心。旧策略已在 `pruning/legacy_strategies.py` 中归档。

## 方法概览

ACGTP 的目标是估计每个 visual token 对动作预测的约束价值，而不是只判断它是否视觉显著。

当前 pipeline：

```text
RGB / depth / robot state / history
    ↓
OpenVLA vision encoder
    ↓
projector visual tokens
    ↓
ACGTP geometry/action constraint scoring
    ↓
constrained token selection
    ↓
post-pruning handoff
    ↓
LLM + action head
```

核心分支：

1. Scene/Layout
   - 支撑面、物体区域、物体边界、障碍/深度边界、scene-aware fill candidate

2. Depth/Structure
   - depth edge、surface discontinuity、object/support boundary

3. Self-Filtered Contact
   - 抑制 gripper self-core，保护 object-side contact ring

4. Motion/Contact Risk
   - 未来运动走廊、swept motion risk、可能接触/碰撞区域

5. Dynamic Controller
   - 根据 phase/risk/confidence 调整 keep ratio 和 branch budget

6. History Stabilizer
   - 用短历史平滑 contact/motion/action scores，避免阶段和保留区域抖动

## 代码结构

核心剪枝代码在 `pruning/`：

| 文件 | 作用 |
|---|---|
| `pruning/hook.py` | projector hook 主入口，负责采集输入、调用策略、返回剪枝后的 visual tokens |
| `pruning/selector.py` | token selector，实现 ACGTP v2/v2-fast 和旧 selector |
| `pruning/strategy_registry.py` | 当前策略注册表，区分 current/baseline/legacy |
| `pruning/legacy_strategies.py` | 历史策略集合，用于 ablation，不作为主线 |
| `pruning/post_pruning.py` | 剪枝后交给 LLM 前的 position ids 与 pruning info 处理 |
| `pruning/static_scene_cache.py` | ACGTP 静态 scene/depth cache |
| `pruning/scene_layout.py` | scene/layout 分支 |
| `pruning/depth_edge.py` | depth edge 与 valid depth mask |
| `pruning/contact_ring.py` | self-filtered contact ring |
| `pruning/motion_corridor.py` | motion corridor 与 motion EMA |
| `pruning/action_constraint.py` | future action constraint score |
| `pruning/acgtp_dynamic_controller.py` | phase/risk/confidence 动态预算控制 |
| `pruning/acgtp_history.py` | ACGTP history buffer 与 branch-wise EMA |
| `pruning/attention_relevance.py` | attention relevance 辅助逻辑 |
| `pruning/metrics.py` | hook/eval 指标结构 |

LIBERO 评测入口：

| 文件 | 作用 |
|---|---|
| `scripts/eval_openvla_baseline.py` | 当前主要 eval 脚本，支持 baseline、depth、ACGTP 策略 |
| `experiments/robot/libero/acgtp_v2_audit.py` | ACGTP v2 审计脚本 |
| `scripts/build_performance_report.py` | 从 run 输出生成性能对照报告 |
| `scripts/action_sensitivity_acgtp_probe.py` | action sensitivity / counterfactual probe |

快捷脚本：

| 文件 | 作用 |
|---|---|
| `run_acgtp_dynamic_fast_075_faststats.sh` | 运行 ACGTP dynamic-fast@0.75 小规模测试 |
| `run_baseline_none_compare_acgtp.sh` | 运行 `none` baseline 对照 |
| `run_step7_faststats_compare.sh` | 运行 Step 7 小规模性能对照 |

## 环境与路径

当前服务器项目路径：

```bash
cd /infini-data/openvla
```

当前 Python 环境：

```bash
/infini-data/miniconda3/envs/openvla/bin/python
```

常用 checkpoint：

```bash
/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial
```

LIBERO 路径：

```bash
/infini-data/LIBERO
```

## 快速验证

### 1. 静态编译

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python -m py_compile \
  pruning/hook.py \
  pruning/selector.py \
  pruning/post_pruning.py \
  pruning/static_scene_cache.py \
  pruning/strategy_registry.py \
  pruning/config.py \
  scripts/eval_openvla_baseline.py
```

### 2. ACGTP v2 审计

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python \
  experiments/robot/libero/acgtp_v2_audit.py --task all
```

期望结果：

- fallback equivalence: PASS
- instruction parser audit: PASS
- semantic backend audit: PASS
- scene-layout branch audit: PASS
- accounting/debug field audit: PASS
- attention alignment/stress: PASS

## 小规模实验

### Baseline: no pruning

```bash
cd /infini-data/openvla
bash run_baseline_none_compare_acgtp.sh
```

### ACGTP dynamic-fast@0.75

```bash
cd /infini-data/openvla
bash run_acgtp_dynamic_fast_075_faststats.sh
```

该 profile 的关键配置：

```bash
--pruning_strategy robot_geo_acgtp_v2
--keep_ratio 0.75
--acgtp_fast_selector_enabled true
--acgtp_full_diagnostics_enabled false
--acgtp_dynamic_enabled true
--acgtp_history_enabled false
--acgtp_v2_semantic_enabled false
--acgtp_v2_semantic_backend none
```

更激进 profile 可在手动实验中调整：

```bash
--keep_ratio 0.58
--acgtp_dynamic_phase_schedule aggressive
--acgtp_dynamic_contact_phase_gate coverage
--acgtp_dynamic_allow_below_base_keep_ratio true
--acgtp_dynamic_min_keep_ratio 0.55
--acgtp_dynamic_max_keep_ratio 0.75
--acgtp_dynamic_branch_floor_enabled true
--acgtp_constrained_fill_max_ratio 0.28
--acgtp_position_preserve_enabled true
```

注意：`0.55` 附近已经观察到任务失败风险，低 keep ratio 需要逐任务验证。

## 性能报告

性能分析不能只看 wall time。当前报告应拆分：

- LLM time
- hook time
- selector time
- CUDA total
- wall time
- actual retention
- success per task

用法示例：

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python scripts/build_performance_report.py \
  --root /infini-data/openvla/outputs/<compare_root> \
  --baseline_method none \
  --output /infini-data/openvla/outputs/<compare_root>/performance_report.md
```

已有报告示例：

```bash
outputs/step7_baseline_vs_acgtp_dynamic_fast_075_20260528_report.md
```

## 输出文件

每个 eval run 通常会在 `outputs/<run_name>/` 下产生：

| 文件 | 说明 |
|---|---|
| `summary.json` | 总体成功率、episode 数、平均指标 |
| `episode_metrics.csv` | 每个 episode 的成功/失败、任务名、步数等 |
| `step_metrics.csv` | 每一步的剪枝、耗时、分支 attribution、fallback 等指标 |
| `videos/` | 可选视频输出，默认小规模性能测试一般关闭 |

重点字段：

| 字段 | 含义 |
|---|---|
| `num_visual_tokens_original` | 原始 visual token 数，通常为 256 |
| `num_visual_tokens_kept` | 剪枝后保留 token 数 |
| `actual_keep_ratio` | 实际保留比例 |
| `hook_total_ms` | hook 整体开销 |
| `selector_total_ms` | selector 选择开销 |
| `llm_forward_ms` | LLM 前向耗时 |
| `cuda_total_ms` | CUDA 统计总耗时 |
| `wall_time_ms` | 端到端 wall time |
| `fallback_used` | 是否进入 fallback |
| `fallback_reason` | fallback 原因 |
| `acgtp_branch_accounting_valid` | 分支 accounting 是否可信 |
| `selected_by_scene_layout_count` | scene/layout 分支保留 token 数 |
| `selected_by_depth_structure_count` | depth/structure 分支保留 token 数 |
| `selected_by_contact_ring_count` | contact 分支保留 token 数 |
| `selected_by_motion_corridor_count` | motion 分支保留 token 数 |
| `selected_by_constrained_fill_count` | constrained fill 保留 token 数 |

## 当前实验判断

当前结论应保守理解：

1. ACGTP 的 pipeline 已经能稳定跑通，depth observation 和 fallback 诊断已修正。
2. dynamic-fast@0.75 的加速不明显，主要原因不是单纯 selector 选择慢，而是 projector-level pruning 对 LLM 内部 KV/cache/attention 路径的压缩不如模型内部剪枝充分。
3. 更激进 keep ratio 可以带来更明显的 LLM token 数下降，但成功率风险上升，需要按任务横向比较。
4. `position_preserve_enabled=True` 当前很重要，关闭后可能导致任务失败或序列位置语义错位。
5. 旧策略如 `robot_geo_branch_budget_v0`、`robot_geo_hybrid_*`、`edge_reserve` 只应作为 ablation/历史对照。

## 推荐实验协议

小规模调参阶段：

```text
LIBERO-Spatial
num_tasks = 4
num_episodes = 2
max_steps = 220
seed = 7
```

比较策略建议只保留：

1. `none`
2. `depth_edge_fast@0.75`
3. `robot_geo_acgtp_v2 dynamic-fast@0.75`
4. 一个更激进的 `robot_geo_acgtp_v2` profile，例如 actual retention 约 0.60

不要同时跑所有 legacy 策略，否则很难判断主线是否收敛。

必须报告：

- total success rate
- per-task success
- LLM time
- hook time
- selector time
- CUDA total
- wall time
- actual retention
- fallback ratio

## 论文主线建议

当前方案建议收敛为：

```text
Action-Constrained Geometric Token Protection for Efficient VLA Inference
```

核心主张：

> VLA token pruning 不应只保留视觉显著 token，而应保护会约束未来机器人动作的几何区域。

推荐理论支撑：

1. Information Bottleneck
   - token budget 是瓶颈，目标是保留对 action prediction 有用的信息。

2. Constrained Coverage
   - scene/depth/contact/motion 可写成动作约束覆盖目标。

3. Counterfactual Action Sensitivity
   - 通过 drop branch/drop region 测动作变化，验证 token 是否 action-relevant。

## 后续路线

优先级从高到低：

1. 继续清理 `selector.py`
   - 将 legacy selector 拆到 `legacy_selector.py`
   - 当前核心 selector 只保留 `acgtp_v2_fast`、`acgtp_v2` 和必要 baseline

2. 分析剪枝后 handoff
   - 对齐 VLA-Pruner / VLA-IAP / VLA-ADP 中剪枝后 position ids、attention mask、cache position 的处理方式
   - 判断是否需要从 projector-level pruning 升级到 model-internal pruning

3. 做 action sensitivity probe
   - drop scene/depth/contact/motion branch
   - 报告 action L2 shift、gripper toggle、jerk、success

4. 设计轻量几何约束专家
   - 先 rule-based ACR
   - 再 Tiny Geometry CNN + Robot-State FiLM residual
   - 不建议现在端到端训练 OpenVLA

5. 做更可靠的多任务小规模实验
   - 控制 run 数
   - 每个任务横向比较成功率
   - 保证加速收益不被 hook/diagnostics 吃掉

## 常见问题

### 为什么剪枝后加速不明显？

当前剪枝发生在 projector 输出之后。它确实减少了传给 LLM 的 visual token 数，但不等同于 VLA-Pruner 这类模型内部剪枝：后者会在指定 LLM layer 内同步更新 hidden states、position ids、attention mask、cache position。当前方案的收益会受到 hook 开销、post-pruning 处理、LLM 实际实现、prefill/decode 结构影响。

### constrained fill 是 fallback 吗？

不是。`constrained_fill` 是正常分支，用于在 scene/depth/action-relevant candidate 内补足预算。真正 fallback 只应在输入缺失、candidate 不足或 selector 失败时出现。

### 缺 depth 时怎么办？

ACGTP 不应伪装正常运行。缺 depth 时应该进入明确的 input fallback，并记录：

```text
fallback_used=True
fallback_reason=missing_depth
```

### 现在可以训练几何专家网络吗？

不建议马上训练。先用规则版 future action constraint score 和 action sensitivity probe 验证策略，再训练极轻量 residual scorer。训练对象最多是剪枝决策模块，不应端到端训练 OpenVLA。

## 原始 OpenVLA 信息

原始项目说明、安装细节和 citation 见：

```bash
oriREADME.md
```

OpenVLA citation：

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal={arXiv preprint arXiv:2406.09246},
    year={2024}
}
```

