# ACGTP 方案重构判断与后续任务 (2026-05-31)

> 目的: 记录对当前 ACGTP 论文主线与工程路线的重新判断。
> 本文是设计整理文档,不代表已经修改核心代码。

## 1. 总判断

ACGTP 不应该继续被表述为"基于机器人几何约束的视觉 token 剪枝"。
这个表述太接近 heuristic rule,也容易与 VLA-IAP 的 interaction-first /
geometric prior 发生正面重合。

更合适的主线是:

> VLA 视觉 token 剪枝不是单一 salience 排序问题,而是机器人执行中多类功能
> token 在固定预算下的竞争问题。ACGTP 通过 execution-function-aware
> quota allocation 显式保护 scene-layout、contact-interaction、
> motion-corridor、semantic/action token,避免 global top-k 误删执行关键结构。

也就是说,ACGTP 的创新点应从"几何约束判断重要性"升级为:

> 把 VLA token pruning 从单一重要性排序改成执行功能感知的结构化预算分配。

这一路线能和当前项目已有发现闭环:早期实验中 robot geometry signal 会和
depth edge / scene layout token 竞争,而 LIBERO-Spatial 又高度依赖 camera-centric
scene layout。失败现象本身说明 global top-k competition 是真实问题,不是简单调参问题。

## 2. 对当前方案的修正定位

当前 ACGTP 已经不应退回"所有几何/深度/夹爪分数加权后 global top-k"。保留现有宪法:

- 不改 OpenVLA 权重。
- 不训练几何专家网络。
- 不加新混合策略名。
- 不双重剪枝。
- 文本 token 永远保留。
- 几何硬保护永不被普通过滤删。
- `P_geo` 超预算时抬高保留率而非删除保护 token。
- fallback 永不静默。
- 每次改动必须 smoke/probe 验证。

但论文级机制应重新拆成执行功能集合:

```text
keep = P_layout union P_contact union P_motion union P_sem union P_act union P_fill
```

其中:

- `P_layout`: scene-layout token。保护柜子、抽屉、容器、桌面、支撑面、目标/参考物边界、
  深度突变和全局空间关系。它解释 LIBERO-Spatial 为什么不能只靠 near-gripper token。
- `P_contact`: contact-interaction token。保护夹爪投影、接触环、目标局部几何、把手、
  可抓取边缘和局部深度结构。
- `P_motion`: motion-corridor token。保护末端执行器未来运动方向、swept volume、
  潜在碰撞/接触路径和可达性约束。
- `P_sem`: instruction semantic token。来自真实 materialized attention 时才叫 true
  semantic attention;在 base OpenVLA + FlashAttention 下只能诚实标注为 QK proxy。
- `P_act`: action execution token。来自 action decode / history attention。不可用时必须
  显式 unavailable,不能用 `P_sem` 冒充。
- `P_fill`: 补齐和空间多样性 token。它不拥有硬保护权。

## 3. 新机制核心: 分支预算保护

后续方法不应再使用所有分数融合后的 global top-k 作为主选择逻辑。建议主流程是:

```text
1. 先加入 geo_protect hard mask。
2. 若 hard mask 超预算,抬高 keep ratio,不删除 hard-protect token。
3. 给 layout/contact/motion/sem/action 分配独立 quota。
4. 各分支内部 top-k。
5. 分支并集。
6. 只对非 hard-protect token 做 redundancy minimization。
7. 用 fill/diversity 补齐固定 token 数。
```

这样 ACGTP 解决的问题变成:

> 单一 salience 排序会让 scene-layout、contact 和 motion token 在同一个 top-k 池中互相挤占,
> 导致关键执行结构被误删。ACGTP 用结构化 quota 显式保留不同执行功能。

这比"根据夹爪距离/深度边缘打分"更像一个可被审稿人接受的机制。

## 4. 阶段自适应不只调 keep ratio

阶段调度应和功能预算联动,而不是只按末端速度调保留率:

| 阶段 | 预算倾向 |
|---|---|
| early / search | `layout` 上调,`semantic` 上调,`contact` 下调,`motion` 中等,keep ratio 中高 |
| approach | `layout` 保底,`motion` 上调,`contact` 上调,`semantic` 中等 |
| contact / grasp / insert | `contact` 大幅上调,local depth edge 上调,`motion` 上调,`layout` 保底且不归零,keep ratio 高 |
| post-contact / stable | 根据风险保留 contact/motion,适当增加 fill/diversity,keep ratio 可降低 |

这个阶段设计需要来自物理/action 证据,不能由低 geometry-attention IoU 单独触发。
这与当前 P3 风险逻辑一致:低 IoU 只能是诊断,不能单独拉高风险。

## 5. 与现有工作的差异定位

| 工作 | 主要思想 | ACGTP 需要突出的差异 |
|---|---|---|
| VLA-Cache | 跨帧静态 token KV 复用 | ACGTP 不是 KV 复用,而是剪枝预算分配与功能保护 |
| VLA-Pruner | semantic attention + action decode attention 双目标 | ACGTP 不依赖真实 attention 默认可得性,用 RGB-D/机器人状态提供低开销执行结构先验;attention 只做候选/验证 |
| VLA-IAP | interaction-first, geometric prior, dynamic scheduling | ACGTP 避免泛泛讲 geometry prior,突出 scene-layout/contact/motion 的预算竞争与分支保护 |
| SAFE-Pruner | future-aware semantic attention cue,缓解浅层 pruning 短视 | ACGTP 用执行结构先验保护低层动作关键区域,不是预测未来层 attention |
| EfficientVLA | 多组件 training-free 加速与压缩 | ACGTP 聚焦 OpenVLA 类视觉 token 的 plug-and-play 保护式剪枝,不动主模型 |
| SpecPrune-VLA | action-aware self-speculative pruning | ACGTP 的阶段调度应联动功能 quota,不只是动作速度门控 |

因此,论文表述应避免:

- "我们引入机器人几何先验来保护重要 token。"
- "根据夹爪距离、运动方向和深度边缘判断 token 重要性。"
- "使用 RGB-D 信息选择关键 token。"

更推荐:

> ACGTP formulates VLA visual-token pruning as an execution-function-aware
> budget allocation problem. Instead of ranking all visual tokens by a single
> salience score, it allocates protected quotas to scene-layout, contact-region,
> motion-corridor, and semantic/action candidates, preventing global top-k
> competition from suppressing manipulation-critical structures.

## 6. 后续工程任务

### P0: 保持核心宪法不变

- 保留 `robot_geo_acgtp_v2` 作为核心策略名。
- 不引入新混合策略名。
- 不改 OpenVLA 权重。
- 不训练 selector。
- 不做 projector + internal 双重剪枝。

### P1: 显式拆分功能分支分数

当前 `geo_soft_score` 更像总排序分数。后续应在 payload 中显式保留:

- `layout_score`
- `contact_score`
- `motion_score`
- `semantic_score` 或 `semantic_proxy_score`
- `action_attention_score` 或 unavailable 状态

`geo_soft_score` 应降级为 fill / tie-breaker,不再承担主选择逻辑。

### P2: quota_config 改成分支预算

把当前偏权重融合的配置升级为功能 quota:

```text
layout_quota_ratio
contact_quota_ratio
motion_quota_ratio
semantic_quota_ratio
action_quota_ratio
fill_quota_ratio
```

预算总和可以按阶段动态归一化,但 hard-protect token 不参与普通预算竞争。

### P3: 阶段自适应 quota 调度

新增或整理 phase 估计:

- early/search
- approach
- contact/grasp/insert
- post-contact/stable

阶段来源应来自 physical/action evidence:

- gripper-object distance / projected contact ring
- action delta / jerk
- motion corridor coverage
- depth validity
- contact evidence coverage

低 attention-geometry IoU 不能单独决定阶段或 high risk。

### P4: 只对非保护 token 做 redundancy minimization

可借鉴 VLA-Pruner 的 combine-then-filter,但过滤范围必须是:

```text
filterable = selected_candidates - geo_protect_mask
final = geo_protect_mask union redundancy_minimize(filterable)
```

禁止删除:

- 文本 token。
- `geo_protect_mask`。
- hard-protect 超预算时的保护 token。

需要记录:

- redundancy 前后各分支 token 数。
- 被过滤的 semantic/action/layout/contact/motion 数。
- hard-protect kept count 是否等于 protected count。

### P5: 第2层 attention 诚实标注

base OpenVLA + FlashAttention 下,当前运行时仍是 QK proxy,不是默认真实 materialized
attention。后续应记录:

- `internal_semantic_attention_source`
- `internal_semantic_attention_is_materialized`
- `internal_action_attention_source`
- `internal_action_attention_available`
- fallback reason

不可用时释放相应 quota 给 fill/diversity 或保守预算,不能静默伪装成 true attention。

## 7. 后续实验命题

### 命题一: VLA pruning 中存在功能 token 竞争

需要量化:

- depth edge token 被 robot/contact token 挤掉的比例。
- layout token 在失败 episode 中的 dropped ratio。
- selected token center bias / near-gripper bias。
- layout/contact/motion 分支的 overlap 和 conflict。

### 命题二: global top-k 会误删关键结构

对比:

- depth-edge global top-k。
- robot-geometry global top-k。
- weighted hybrid global top-k。
- quota-based functional allocation。

重点不是只比较速度,而是比较 success、critical deletion、layout coverage、contact coverage。

### 命题三: quota allocation 不是偶然调参

必须做 ablation:

- w/o layout quota。
- w/o contact quota。
- w/o motion quota。
- w/o fill/diversity。
- global top-k instead of quota。
- fixed quota vs phase-adaptive quota。

### 命题四: 指标继续对齐论文

至少报告:

- success rate。
- episode steps。
- model forward time。
- prefill time / decode time。
- hook overhead。
- effective inference time = model forward + hook。
- token retention。
- speedup。
- GPU memory。
- theory-vs-wall gap。

base OpenVLA 仍应诚实标注 decode floor:base 只证明选择质量和机制有效性,
端到端加速兑现留给 OFT/pi0 或其他 prefill-dominated 模型。

## 8. 近期最小落地路线

1. 写一个只读 probe,从现有 payload 中重构 layout/contact/motion 三分支分数,
   统计 global top-k competition。
2. 不改 forward,先在离线 probe 里模拟 quota allocation,输出每分支保留数、
   overlap、protected coverage、layout/contact/motion coverage。
3. 若离线结果显示 quota allocation 能减少 layout/contact 互相挤占,再改
   `resolve_visual_keep_indices` 的 `geo_guarded` 分支。
4. 改代码后跑:

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python -m py_compile \
  pruning/internal_pruning.py pruning/hook.py

/infini-data/miniconda3/envs/openvla/bin/python scripts/probe_attention_proxy_gap.py \
  --prune_layer 2 --keep_ratio 0.50 --output_dir outputs/attention_proxy_gap

/infini-data/miniconda3/envs/openvla/bin/python scripts/run_core_surface_validation.py \
  --output_root outputs/function_quota_smoke \
  --num_tasks 1 --num_episodes 1 --max_steps 60 \
  --methods internal_geo_guarded_050,internal_dynamic_050
```

通过标准:

- `internal_pruned_geo_critical_count=0`。
- hard-protect kept count 等于 protected count。
- fallback 显式记录。
- 低 IoU 不单独触发 high risk。
- layout/contact/motion 分支覆盖率可解释。

## 9. 一句话版本

ACGTP 不需要换方向,需要换论文级表达和预算机制粒度。最强版本不是"几何剪枝器",
而是:

> 面向机器人操作执行的 function-aware structured token allocation:
> 用显式分支预算保护 scene-layout、contact-interaction、motion-corridor、
> semantic/action candidates,避免 global top-k competition 误删执行关键结构。
