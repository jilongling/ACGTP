# ACGTP 第2层语义 token 保护修正方案 (2026-05-30)

> 范围: 先读懂前情并形成方案,不修改核心剪枝代码。
> 当前结论基于 `docs/gap1_attention_verification_wip_20260530.md`,
> `docs/acgtp_status_20260530.md`, `docs/vla_efficient_inference_survey_20260530.md`,
> `docs/prefill_retention_sweep_20260530.md`,
> `docs/metric_alignment_report_20260530.md`, 项目记忆
> `acgtp-final-design.md` / `acgtp-progress.md`, 以及
> `pruning/internal_pruning.py`、`pruning/hook.py` 的当前实现。

## 0. 本次处理结论

1. ACGTP 第2层的宪法目标仍应保持为: 全 256 个视觉 token 进入 LLM 前 K 层后,
   用 LLM 内部跨模态信号生成 `P_sem` 和 `P_act`, 再与 `P_geo`、`P_fill`
   做配额并集。attention 只能提供候选和保护证据,不能替代几何硬保护。
2. 当前 base OpenVLA + FlashAttention 运行时,第2层实际使用的是
   QK text-to-vision proxy,不是默认可得的真实物化 attention。代码路径是:
   `hook.py:820-846` 的 `quota_config` 没有 `try_output_attentions`,导致
   `internal_pruning.py:993` 默认为 false,随后在 `1014-1020` 回退到
   `_qk_text_to_visual_attention`。
3. VLA-Cache 和 VLA-Pruner 都不支持"语义 attention 一票决定删除"。它们的可借鉴点是:
   attention 用作任务相关 token 的保护阈值或候选池信号,最终还要和动作/几何/冗余机制分开处理。
4. 现有只读 probe 已有有效产物:
   `outputs/attention_proxy_gap/attention_proxy_gap.json`。结果显示
   `seq_len_at_prune_layer=291`, runtime source 为 `llm_qk_text_to_vision`;
   QK proxy vs real eager attention 的 top-k IoU 为 `0.4463`, Spearman 为 `-0.2743`;
   real attention vs geometry 的 top-k IoU 为 `0.148`;
   real attention 对 `geo_protect_mask` 的 top-k 覆盖为 `0.0`。
   这说明真实语义 attention 与几何保护高度不重合,低 IoU 必须继续只作为诊断信号,
   不能触发高风险,更不能允许 attention 删除几何硬保护。

## 1. 第1步: 复述确认理解

### 1.1 ACGTP 当前第2层实现

设计宪法里的第2层是 "True LLM Attention Verification": 全量视觉 token 先进入
LLM 跑到 `prune_layer`。之后读取真实 LLM 内部注意力,生成:

- `P_sem`: 指令语义相关的视觉 token,对应 text/instruction 到 vision 的相关性。
- `P_act`: 历史动作或 action decode 对视觉 token 的执行相关性。

最终选择不是全局加权 top-k,而是:

```text
keep = P_geo union P_sem union P_act union P_fill
```

当前代码已经实现了这个"配额并集"的骨架:

- `hook.py:848-909` 生成 `geo_protect_mask` 和 `geo_soft_score`。
- `internal_pruning.py:708-721` 先加入显式 `geo_protect_mask`,超预算则抬高
  `target_k`,不丢保护 token。
- `internal_pruning.py:752-762` 再按独立配额加入 semantic attention 和 historical/action
  attention。
- `internal_pruning.py:764-777` 最后用 fill/fallback 补齐。
- `internal_pruning.py:779-838` 组装 keep set 和诊断字段,其中
  `internal_pruned_geo_critical_count` 已基于显式 hard mask。

但第2层的 attention 来源还不是宪法要求的完整真实 attention。当前 `quota_config`
没有 `try_output_attentions`,所以 `internal_pruning.py:993` 的
`try_materialized_attn` 默认为 false。运行时不会要求 prune layer 输出 attention
matrix,因此 `layer_attn` 通常为 None,再由 `_qk_text_to_visual_attention`
用该层真实 Q/K 投影和当前 hidden states 算一个 text-to-vision proxy。

所以当前更诚实的表述是:

```text
P_sem = LLM QK proxy text-to-vision candidate, not materialized LLM attention
P_act = action/decode attention history if explicitly captured and available, otherwise unavailable
```

### 1.2 VLA-Cache 的语义 attention 用法

VLA-Cache 的 attention 不是"保留打分器",而是"保护阈值"。它先找跨帧静态且可复用的 token,
再用 decoder text-to-vision cross-attention 判断哪些静态 token 仍然任务相关。
高于阈值的 token 从可复用集合里剔除,强制每步重算。也就是说:

- attention 的角色是保护任务相关 token,防止它们被缓存复用掉。
- 它服务于 KV 复用机制,不是视觉 token 剪枝的全局 top-k。
- 它没有替代几何,也没有解决机器人接触/把手/夹爪末端这类低层执行细节的保护问题。

对 ACGTP 的可借鉴点是: 语义 attention 可以作为"不要轻易删/复用"的保护证据。
一致之处是它也把 attention 放在保护侧,而不是让 attention 吞掉全部选择权。
潜在冲突是: VLA-Cache 的核心目标是跨帧 KV 复用,不等同于 ACGTP 当前的中间层缩序列剪枝;
不能把它的 cache policy 直接搬成 ACGTP 的剪枝 policy。

### 1.3 VLA-Pruner 的语义 attention 用法

VLA-Pruner 更接近 ACGTP 第2层: 它把 prefill 语义 attention 和 action decode attention
分成两个候选信号。

- `S_vl[m] = mean_i A_vl[i,m]`: prefill 阶段的语义相关性,取 `C_vl=Top-M(S_vl)`。
- `S_act`: action decode 阶段的执行精度相关性,取 `C_act=Top-M(S_act)`。
- `C_dual = C_vl union C_act`,之后再做 redundancy minimization。

这和 ACGTP 的 `P_sem union P_act` 哲学一致。它还明确指出语义 attention 偏高层目标物
和语言区域,会漏掉夹爪末端、接触边界、把手等低层执行细节。这正好支持 ACGTP 的几何护栏:
语义 attention 不能覆盖几何,低几何-attention IoU 不能自动变成高风险。

可借鉴点:

- 语义 attention 只做一路候选池,不是最终裁判。
- prefill 语义和 action decode 执行信号要分开建模,再做并集。
- 如果引入冗余过滤,也应在候选并集之后做,且必须排除几何硬保护。

潜在冲突:

- 如果 redundancy minimization 被放在 `geo_protect_mask` 之后但仍允许删除 hard-protect token,
  就违反 ACGTP 宪法。
- 如果把 `S_vl` 当作全局 top-k 或风险触发器,会重新落回"语义 attention 单信号剪枝"的问题。

## 2. 第2步: 修正/完善第2层语义 token 保护方案

### 2.1 语义信号来源: 真实 attention、prefill VL attention 与 QK proxy

base OpenVLA 当前使用 FlashAttention,默认不物化 attention probabilities。
因此不能在生产路径里默认宣称拿到了 VLA-Pruner 式真实 prefill VL attention。

建议分成三层诚实来源:

1. **默认 base 路径: QK proxy,但必须显式标注。**
   继续使用 `_qk_text_to_visual_attention` 作为 `P_sem_proxy`,因为它至少来自真实 LLM
   layer 的 Q/K 投影和当前 hidden states,不是 hook 侧几何分数。但所有报告字段和文档
   都应把它标为 `llm_qk_text_to_vision`,不能写成 true materialized attention。

2. **只读验证路径: eager/sdpa 重放 probe。**
   继续用 `scripts/probe_attention_proxy_gap.py` 定量衡量 QK proxy 与真实物化 attention
   的差距。当前有效产物已经说明 QK proxy 与 real attention 差距不小,后续每次动第2层都要重跑:

   ```bash
   cd /infini-data/openvla
   /infini-data/miniconda3/envs/openvla/bin/python scripts/probe_attention_proxy_gap.py \
     --prune_layer 2 --keep_ratio 0.50 --output_dir outputs/attention_proxy_gap
   ```

   判读底线:

   - `seq_len_at_prune_layer` 必须是 prefill 长度,当前应为 `291`,不能退回 `1`。
   - `internal_attention_source_runtime` 若仍是 `llm_qk_text_to_vision`,报告必须继续叫 proxy。
   - `qk_proxy__vs__real_attn` 的 IoU/Spearman 用于衡量代理误差,不直接作为成功风险。
   - `geo_protect_topk_coverage` 低时,结论是 attention 会漏几何保护,不是几何应让位。

3. **未来真实 attention 路径: 仅在模型/后端可承受时启用。**
   如果未来切到 OFT/pi0/eager/sdpa 或只在 prune layer 正确重放 eager attention,
   可以把 VLA-Pruner 式 prefill `S_vl` 作为真正 `P_sem`。但这应是显式 opt-in,
   并且记录代价: 显存、速度、attention_mask/RoPE 重建正确性。base+FlashAttention 下不应
   静默切换。

### 2.2 语义 attention 在配额并集里的角色

`P_sem` 应保持为候选池的一路信号:

```text
P_geo first, hard protected
P_sem next, semantic quota
P_act next, action quota
P_fill last, residual fill
```

约束:

- `geo_protect_mask` 永远先加入。若 `P_geo` 超预算,抬高 `target_k`,不删除保护 token。
- `P_sem` 只占自己的 semantic quota,不能吃掉 `P_geo`。
- `P_sem` 与 `P_geo` 的低 IoU 只能写入诊断字段,不能单独触发 high risk。
- 如果 materialized attention 不可得,要么使用已标注的 QK proxy,要么释放 semantic quota
  给 fill/fallback,不能静默伪装成真实 attention。
- 当前 `attention_requires_geometry_alignment` 可以作为安全门,但它不应变成"语义必须服从
  geo_soft_score 的全局排序"。语义候选的价值正是补几何看不到的目标语义区域。

当前代码的 P3 风险逻辑已经符合这条原则: attention disagreement 只是有上限的小 bonus,
且只有 physical risk 已升高时才进入风险,不能单独把 keep ratio 拉高。后续不能破坏这一点。

### 2.3 语义 + 动作双目标

建议把第2层拆成两个明确、可记录来源的集合:

#### P_sem: prefill instruction/text -> vision

- 目标: 保护任务语义相关 token,例如目标物、参考物、语言描述区域。
- 优先真实来源: prune layer 的 materialized prefill text-to-vision attention。
- base 默认来源: QK proxy,字段名和报告中标为 proxy。
- 验证: 与 real eager replay 的差距通过 `probe_attention_proxy_gap.py` 量化。

#### P_act: action decode / history -> vision

- 目标: 保护执行细节,例如夹爪末端、接触边界、把手、放置支撑面和短期动作相关区域。
- 优先真实来源: 已完成动作 decode 步的 action-to-vision attention,按时间 EMA 平滑,
  下一帧 prefill 使用历史 `S_act`。这与 VLA-Pruner 对 action decode attention 的用法一致。
- base FlashAttention 下若 decode attention 不物化,`P_act` 必须标为 unavailable。
  不能用 `P_sem` 冒充 `P_act`。
- hook 侧的 `action_constraint_scores` 属于 `P_geo`/物理几何证据,不是 LLM action attention。
  可以与 `P_act` 互补,但不要混名。

现有代码中 `action_attention_history` 已有容器,`_record_decode_attention` 也有将 decode
attention 映射回原视觉 token 的路径。但默认 `capture_decode_attention` 为 false,而且
FlashAttention 下 output attentions 仍可能不可得。因此当前应把 `P_act` 状态写清楚:
"available / unavailable / source / history_length",并让 unavailable 显式释放 quota。

### 2.4 是否引入 VLA-Pruner 式 redundancy minimization

建议: 可以作为后续增强,但不作为修缺口①的第一步。

引入条件:

- 先完成 attention 来源诚实标注和 proxy-gap probe。
- 先保证 `P_geo` hard-protect、`P_sem`、`P_act` 的集合边界清楚。
- 先有 smoke/probe 证明当前 union 不破坏 `geo_critical=0`。

若引入,过滤位置应是:

```text
raw_candidates = P_geo_hard union P_geo_quota union P_sem union P_act union P_fill_candidates
protected = P_geo_hard
filterable = raw_candidates - protected
final = protected union redundancy_minimize(filterable)
```

过滤规则:

- 绝不处理文本 token。
- 绝不删除 `geo_protect_mask`。
- 如果 `protected` 已超过预算,抬高保留率,不做硬保护裁剪。
- redundancy filter 只在非保护集合中做多样性/mRMR,例如在空间邻域、patch embedding 相似度
  或 attention 相似度上去冗余。
- 如果 redundancy filter 导致 `P_sem` 或 `P_act` 被大量挤掉,必须写入诊断字段,不能静默。

这样可以借鉴 VLA-Pruner 的 combine-then-filter,同时不和 ACGTP 的几何护栏打架。

### 2.5 建议的实施顺序

1. **文档与报告口径先收敛。**
   把第2层在 base 上称为 `QK proxy` 或 `P_sem_proxy`,不要称为 true attention。
   当前本文件就是这一步。

2. **探针复跑确认缺口①。**
   使用固定命令重跑 `probe_attention_proxy_gap.py`,记录
   QK proxy vs real attention、real attention vs geometry、geo_protect coverage。
   若结果继续显示 real attention 不覆盖 `geo_protect_mask`,则进一步证明几何硬保护不可被删。

3. **小改 metadata,不改选择语义。**
   如果后续要动代码,第一步只加字段:
   `internal_semantic_attention_source`,
   `internal_semantic_attention_is_materialized`,
   `internal_action_attention_source`,
   `internal_action_attention_available`。
   不改变 keep set,只让报告更诚实。

4. **可选 opt-in true attention path。**
   只在明确配置下尝试 `try_output_attentions`,并在失败时显式 fallback 到 QK proxy。
   fallback 字段必须写清楚,不能静默。

5. **P_act 历史执行 attention 单独验证。**
   若要启用 `capture_decode_attention`,先用 probe 检查 decode attention 是否真的可用,
   是否能正确映射回原 256 视觉 token。不可用时保持 quota release。

6. **最后再考虑 redundancy minimization。**
   它是优化候选冗余,不是修复 attention 来源的前置条件。

### 2.6 验证要求

任何后续核心代码改动后,至少跑:

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python -m py_compile \
  pruning/internal_pruning.py pruning/hook.py scripts/probe_attention_proxy_gap.py

/infini-data/miniconda3/envs/openvla/bin/python scripts/probe_attention_proxy_gap.py \
  --prune_layer 2 --keep_ratio 0.50 --output_dir outputs/attention_proxy_gap

/infini-data/miniconda3/envs/openvla/bin/python scripts/run_core_surface_validation.py \
  --output_root outputs/layer2_semantic_smoke \
  --num_tasks 1 --num_episodes 1 --max_steps 60 \
  --methods internal_geo_guarded_050,internal_dynamic_050
```

通过标准:

- probe 捕获 prefill: `seq_len_at_prune_layer=291`。
- attention source 和 fallback 被显式记录。
- `internal_pruned_geo_critical_count=0`。
- `internal_geo_explicit_protected_kept_count == internal_geo_explicit_protected_count`。
- 低 `geo_attention_iou` 不会单独触发 high risk。
- smoke 运行完成,核心方法不新增 silent fallback。

## 3. 本次实际做了什么

- 按顺序阅读了 5 份当前文档、2 份项目记忆和关键实现。
- 确认当前第2层运行时仍是 QK proxy,不是默认物化 attention。
- 确认 `geo_protect_mask` 的 hard-protect 路径和配额并集骨架仍在。
- 读取了现有只读 probe 产物,发现 gap-1 probe 已有有效结果,可用于方案判断。
- 本次没有修改 `pruning/internal_pruning.py`、`pruning/hook.py` 或任何核心模型/剪枝代码。

## 4. 回报摘要

第2层的修正方向不是把 VLA-Cache/VLA-Pruner 的 attention 规则硬搬进 ACGTP,
而是把它们的共同原则吸收进 ACGTP 宪法:

- attention 是保护/候选信号,不是几何的替代品。
- semantic 和 action 必须分开,再并集。
- QK proxy 可以作为 base 阶段的实用 fallback,但必须诚实命名和量化误差。
- redundancy minimization 只能过滤非 hard-protect 候选,不能碰 `geo_protect_mask`。
- base 上的第2层结论应表述为"selection quality + mechanism validity + proxy-gap quantified",
  真实 attention 和端到端速度兑现留给可物化 attention / prefill 主导的后续模型路径。
