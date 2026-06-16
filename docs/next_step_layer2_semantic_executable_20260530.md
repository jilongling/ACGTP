# 下一步执行提示词:完善 ACGTP 第2层语义 token 保护 (2026-05-30)

> 给新对话的可执行指令。基于 Codex 报告
> `docs/acgtp_layer2_semantic_attention_design_20260530.md` 的方案,
> 转成具体改动步骤 + 验证命令。

## 背景(1 分钟速读)

你在 /infini-data/openvla 开发 ACGTP(即插即用 VLA 视觉 token 剪枝)。
方案体检发现**缺口①(最大)**:设计宪法第2层要"真实 LLM 注意力",但代码实际用
QK 投影代理(`_qk_text_to_visual_attention`),不是物化注意力。根因:模型跑
FlashAttention,真实注意力矩阵默认不可得;`quota_config` 没有 `try_output_attentions`。

Codex 已给方案(见上述报告),核心结论:
- base+FlashAttention 下继续用 QK proxy,但**必须诚实标注**,不能称 true attention。
- 借鉴 VLA-Pruner:语义 attention 只做候选池一路,不替代几何;P_sem(语义)和
  P_act(动作执行)分开建模再并集;低 geo-attention IoU 只做诊断,不触发高风险。
- 探针已跑出有效结果(`outputs/attention_proxy_gap/attention_proxy_gap.json`):
  QK proxy vs real attention IoU=0.4463,real attention 对 geo_protect 覆盖=0.0
  → 证明语义 attention 不能覆盖几何硬保护。

## 你要做的事(按顺序,每步等我确认再下一步)

### 第1步:复跑探针,确认缺口①的定量证据(纯只读,5 分钟)

**目的**:Codex 报告引用的探针结果可能是旧的(有 bug 的那次)。重跑一次,
拿到干净的 QK proxy vs real attention vs geometry 三向对比数据,作为后续改动的基线。

**命令**(必须用 openvla env):
```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python scripts/probe_attention_proxy_gap.py \
  --prune_layer 2 --keep_ratio 0.50 --output_dir outputs/attention_proxy_gap
```

**验证点**(读 `outputs/attention_proxy_gap/attention_proxy_gap.json`):
- `seq_len_at_prune_layer` 必须是 291(prefill),不能是 1(decode 步,说明 bug 没修好)。
- `internal_attention_source_runtime` 应为 `llm_qk_text_to_vision`(确认当前用代理)。
- `topk_iou` 和 `spearman` 应非空(三向对比数据齐全)。
- 重点看:
  - `topk_iou["qk_proxy__vs__real_attn"]`:代理与真实注意力的 top-k 重合度。
  - `topk_iou["real_attn__vs__geometry"]`:真实注意力与几何的重合度(预期很低)。
  - `geo_protect_topk_coverage["real_attn"]`:真实注意力对几何硬保护的覆盖(预期接近 0)。

**跑完后贴结果给我,我来解读"代理误差多大、真实注意力是否覆盖几何"。**

---

### 第2步:诚实标注当前是 QK proxy,不改选择逻辑(小改 metadata,1 小时)

**目的**:让报告/字段明确写"当前用 QK proxy",不能称 true attention。
这是 Codex 方案 2.5 的第3步"小改 metadata,不改选择语义"。

**要改的文件**:
1. `pruning/metrics.py`:在 `HookMetrics` dataclass 加字段(约 900 行附近):
   ```python
   internal_semantic_attention_source: Optional[str] = None  # "qk_proxy" | "materialized" | "unavailable"
   internal_semantic_attention_is_materialized: Optional[bool] = None
   internal_action_attention_source: Optional[str] = None  # "decode_history" | "unavailable"
   internal_action_attention_available: Optional[bool] = None
   ```
   并在 `to_eval_stats` 和 `to_fast_eval_stats` 里输出这4个字段(约 1139 行和 1161 行附近)。

2. `pruning/internal_pruning.py`:在 `resolve_visual_keep_indices` 的 info dict 里
   (约 799-837 行)填这4个字段:
   ```python
   info.update({
       "internal_semantic_attention_source": "qk_proxy" if (sem_available and attention_source_name == "llm_qk_text_to_vision") else ("materialized" if sem_available else "unavailable"),
       "internal_semantic_attention_is_materialized": bool(sem_available and attention_source_name != "llm_qk_text_to_vision"),
       "internal_action_attention_source": hist_source if hist_available else "unavailable",
       "internal_action_attention_available": bool(hist_available),
   })
   ```

3. `docs/acgtp_status_20260530.md`:在"当前状态"节明确写:
   > 第2层当前使用 QK 投影代理(`_qk_text_to_visual_attention`),不是物化注意力。
   > 根因:FlashAttention 不物化 attention 概率,`quota_config` 无 `try_output_attentions`。
   > 代理与真实注意力的差距已通过探针量化(见 gap1 交接文档)。

**验证**(smoke,10 分钟):
```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python -m py_compile pruning/metrics.py pruning/internal_pruning.py

# 跑 1 task 1 ep 确认新字段出现在 step_metrics.csv
/infini-data/miniconda3/envs/openvla/bin/python scripts/eval_openvla_baseline.py \
  --pretrained_checkpoint /infini-data/checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial --center_crop true \
  --run_root_dir outputs/layer2_metadata_smoke \
  --num_tasks 1 --num_episodes 1 --max_steps 60 \
  --acgtp_compression_backend internal --acgtp_internal_pruning_enabled true \
  --acgtp_internal_selection_mode geo_guarded --keep_ratio 0.50

# 确认新字段
head -1 outputs/layer2_metadata_smoke/*/step_metrics.csv | grep -oE "internal_semantic_attention_source|internal_action_attention"
```

**通过标准**:
- py_compile OK。
- step_metrics.csv 有4个新列。
- `internal_semantic_attention_source` 应为 `qk_proxy`(不是 materialized)。
- `internal_pruned_geo_critical_count` 仍为 0(没破坏几何硬保护)。

**改完后告诉我,我来确认是否进第3步。**

---

### 第3步(可选):opt-in 真实物化 attention 路径(核心改动,需你明确批准)

**目的**:给未来 eager/sdpa 模型或 OFT 留一条真实 attention 路径。
但 base+FlashAttention 下默认仍用 QK proxy,不强制切换。

**要改的地方**:
1. `pruning/hook.py:820-846` 的 `quota_config` 构造里加:
   ```python
   "try_output_attentions": bool(getattr(self.config, "acgtp_internal_try_materialized_attention", False)),
   ```
2. `pruning/config.py` 的 `PruningHookConfig` dataclass 加字段(约 220 行):
   ```python
   acgtp_internal_try_materialized_attention: bool = False
   ```
   并在 `from_eval_cfg` 里解析(约 490 行):
   ```python
   acgtp_internal_try_materialized_attention=_as_bool(cfg.get("acgtp_internal_try_materialized_attention", False)),
   ```
3. `pruning/internal_pruning.py:1014-1020`:在 QK proxy fallback 前加日志:
   ```python
   if capture_for_pruning and attention_for_selection is None:
       # FlashAttention or try_output_attentions=False: fallback to QK proxy
       attention_for_selection = _qk_text_to_visual_attention(...)
   ```

**这一步触及核心 forward 逻辑,需你明确说"可以动"再做。**
而且即使做了,base 上默认仍是 QK proxy(flag 默认 False)。

---

### 第4步(未来):P_act 历史动作 attention + redundancy minimization

Codex 方案 2.3/2.4 提到的两个增强,但**不是修缺口①的前置条件**。
等第2步完成、探针结果解读清楚、你确认"QK proxy 可接受"后,再决定要不要做。

---

## 立即可做 vs 需等你确认

| 步骤 | 类型 | 是否需你批准 |
|---|---|---|
| 第1步(复跑探针)| 纯只读 | **立即可做**,跑完贴结果 |
| 第2步(诚实标注 metadata)| 小改字段,不改选择 | 可做,但建议你看一眼改动再确认 |
| 第3步(opt-in 真实 attention)| 核心改动 | **必须你明确批准** |
| 第4步(P_act + 去冗余)| 未来增强 | 暂不做,等前面完成 |

## 我的建议

**先做第1步(复跑探针),立即可执行,5 分钟出结果。** 我拿到干净数据后能告诉你:
- QK 代理误差到底多大(IoU/Spearman)。
- 真实注意力是否覆盖几何硬保护(预期不覆盖 → 证明几何不可被 attention 删)。
- 当前 QK proxy 在 geo_guarded 里实际起了什么作用(是纯摆设还是真选了语义 token)。

有了这个定量基线,第2步(诚实标注)才有数据支撑,第3步(要不要上真实 attention)
才能理性决策。

**要我现在就帮你跑第1步吗?还是你想先看 Codex 报告、自己手动跑?**
