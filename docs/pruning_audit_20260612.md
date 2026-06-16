# OpenVLA Visual Token Pruning 代码审计报告

> 审计时间: 2026-06-12
> 审计范围: `pruning/` 全部 Python 文件 + `scripts/eval_openvla_baseline.py`
> 核心版本: `functional_quota_static_050_layer2` 分支

---

## 一、目录结构

```
pruning/
├── __init__.py
├── _compat.py
├── config.py                  # PruningHookConfig 数据类，~280 个字段
├── hook.py                     # VisualTokenPruningHook 主类
├── method_profiles.py
├── runtime_config.py           # 配置归一化、geometry/pruning 开关
├── strategy_registry.py        # 策略注册与校验
├── core/
│   ├── __init__.py
│   ├── metrics.py              # HookMetrics (~1000 个字段)
│   ├── utils.py
│   └── visualization.py
├── internal/
│   ├── __init__.py
│   ├── backend.py              # ACGTPInternalPruningBackend，~1820 行
│   ├── quota_config.py         # build_internal_quota_config
│   └── uniform.py
├── legacy/
│   ├── __init__.py
│   ├── acgtp_v1.py
│   ├── branch_budget.py
│   ├── hybrid.py
│   ├── runtime.py              # HookLegacyRuntimeMixin
│   └── strategies.py
├── methods/
│   ├── __init__.py
│   ├── acgtp_v2.py            # select_acgtp_v2_fast (ACGTP-v2 选择器)
│   ├── baselines.py
│   ├── functional_quota.py     # select_internal_quota_tokens (functional quota)
│   ├── registry.py
│   └── utils.py
├── runtime/
│   ├── __init__.py
│   ├── diagnostics.py
│   ├── fast.py                # HookFastRuntimeMixin，~1230 行
│   ├── geometry.py             # HookGeometryMixin
│   └── post.py               # PostPruningStateManager
├── signals/
│   ├── __init__.py
│   ├── action.py              # compute_future_action_constraint_scores, motion corridor
│   ├── robot.py               # RobotState, TokenGeometryCache, ACGTPStaticSceneCache
│   ├── semantic.py
│   ├── spatial.py             # compute_contact_ring_scores, compute_depth_edge_scores
│   └── temporal.py            # GeometryHistoryBuffer, ACGTPHistoryBuffer
└── tests/
    ├── __init__.py
    ├── behavior_regression.py
    └── validation_tests.py
```

---

## 二、数据流：完整端到端路径

### 2.1 两条压缩后端（compression_backend）

```
config.py: acgtp_compression_backend = "projector" | "internal"
           acgtp_internal_pruning_enabled = True/False
```

#### 路径 A: `projector` 后端（传统剪枝）

```
pixel_values
  → vision_encoder (ViT)              [B, 256, D_vision]
  → projector (MLP)                  [B, 256, D_hidden]
  → _projector_hook (hook.py:507)    ← 注册在 projector 上
      → _run() (HookFastRuntimeMixin)
          → _run_acgtp_fast_runtime (fast.py:647)
              1. 从 TokenGeometryCache 获取 depth / camera intrinsics
              2. 采样 token depth → valid_mask
              3. project_tokens_to_robot → 3D 坐标
              4. compute_depth_edge_scores (spatial.py)
              5. compute_scene_layout_scores (spatial.py)
              6. compute_contact_ring_scores (spatial.py)
              7. compute_motion_corridor_scores (action.py)
              8. compute_future_action_constraint_scores (action.py)
              9. select_acgtp_v2_fast (methods/acgtp_v2.py)
              10. _finalize_acgtp_fast_runtime (fast.py:52)
                  → visual_tokens.index_select(dim=1, index=keep_indices)  ← 实际剪枝
                  → _prepare_position_preserve_info (post.py:43)
                      → PostPruningStateManager 写入 _pending_position_info
  → LLM input: [B, K+text, D_hidden]  (K = keep_count)
  → concatenate with action tokens
  → LLM.forward(position_ids=preserved)  ← _language_model_pre_hook 注入 position_ids
```

#### 路径 B: `internal` 后端（Layer-2 剪枝）

```
pixel_values
  → vision_encoder (ViT)              [B, 256, D_vision]
  → projector (MLP)                   [B, 256, D_hidden]
  → _projector_hook (hook.py:507)     ← 注册在 projector 上
      → _run() (HookFastRuntimeMixin)
          → _run_acgtp_fast_runtime (fast.py:647)
              同上步骤 1-9
              10. _finalize_acgtp_fast_runtime (fast.py:52)
                  → _prepare_internal_pruning_plan (hook.py:313)
                      → PostPruningStateManager.prepare_internal_pruning_plan
                          → backend.set_pending_plan(InternalPruningPlan)
                              携带 geometry_payload、quota_config、keep_indices
                  → pruned = visual_tokens  ← 不做剪枝，返回原始 tensor
  → LLM input: [B, 257, D_hidden]  (完整 projector 输出)
  → LLM.forward (patched, backend.py:1380)
      → layers 0, 1: 正常 forward，hidden_states 完整 [B, 257, 4096]
      → layer 2:
          → resolve_visual_keep_indices (backend.py:873)
              → 从 geometry_payload 提取 scene/depth/contact/motion scores
              → select_internal_quota_tokens (functional_quota.py)
                  分配 quota: layout(30%) / contact(20%) / motion(15%) / sem(12%) / action(8%) / fill(15%)
              → 返回新的 keep_indices (可能与 hook 侧不同)
          → hidden_states = hidden_states.index_select(1, keep_indices)  ← 实际剪枝
          → position_ids = keep_indices.unsqueeze(0)
          → cache_position = keep_indices
          → causal_mask 重算
      → layers 3-31: forward([B, K+1, D_hidden])
```

### 2.2 Hook 注册顺序（attach_to_model, hook.py:503-559）

```python
# Step 1: projector forward hook（两个后端都注册）
for name, module in model.named_modules():
    if name == "projector":
        self._hook_handle = module.register_forward_hook(self._projector_hook)

# Step 2: internal 后端注册（仅 internal 模式）
if self._internal_pruning_requested():
    self._internal_backend = enable_acgtp_internal_pruning(model, ...)

# Step 3: LLM pre-hook（仅 projector 模式，需要 position preserve）
if language_model is not None and not self._internal_pruning_requested():
    self._lm_pre_hook_handle = language_model.register_forward_pre_hook(
        self._language_model_pre_hook, with_kwargs=True
    )
```

### 2.3 Detach 顺序（hook.py:561-570）

```python
def detach(self):
    self._hook_handle.remove()           # 1. projector hook
    self._lm_pre_hook_handle.remove()  # 2. LLM pre-hook
    disable_acgtp_internal_pruning(...)  # 3. internal backend
```

---

## 三、各模块详细分析

### 3.1 `config.py` — PruningHookConfig

**规模**: ~280 个字段，`from_eval_cfg()` ~570 行

**关键字段**:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `strategy` | `"none"` | 策略名称 |
| `keep_ratio` | `1.0` | 保留比例 |
| `acgtp_compression_backend` | `"projector"` | 压缩后端 |
| `acgtp_internal_pruning_enabled` | `False` | 启用 internal 后端 |
| `acgtp_internal_prune_layer` | `2` | internal 剪枝层 |
| `acgtp_fast_selector_enabled` | `True` | ACGTP-v2 fast selector |
| `acgtp_dynamic_enabled` | `True` | 动态相位控制 |
| `acgtp_static_scene_cache_enabled` | `True` | 静态场景缓存 |
| `acgtp_latency_plan_cache_enabled` | `False` | 延迟计划缓存 |
| `acgtp_history_enabled` | `False` | 历史稳定器 |
| `acgtp_attention_guidance_enabled` | `False` | 注意力引导 |
| `acgtp_internal_functional_quota_enabled` | `True` | functional quota |
| `acgtp_internal_selection_mode` | `"geo_guarded"` | internal 选择模式 |

**Quota 分配** (functional_quota):
```
layout_quota_ratio:      0.30  (场景结构)
contact_quota_ratio:    0.20  (接触区)
motion_quota_ratio:     0.15  (运动走廊)
semantic_quota_ratio:   0.12  (语义/注意力)
action_quota_ratio:     0.08  (动作约束)
fill_quota_ratio:       0.15  (填充)
→ 总和 = 1.00
```

**从 YAML/CLI 配置读取**: 通过 `from_eval_cfg(cfg)` 批量映射，支持别名（如 `pruning_mode`/`pruning_method`/`pruning_strategy` 均映射到 `strategy`）。

**潜在问题**:
- **配置字段过多**: ~280 个字段，难以维护。大量字段从未被使用或仅在特定分支生效。
- **配置合法性依赖运行时检查**: `from_eval_cfg` 不验证互斥条件（如 `acgtp_compression_backend="projector"` 但 `acgtp_internal_pruning_enabled=True`）。
- **quota 总和未验证**: `acgtp_internal_*_quota_ratio` 6 个字段加起来应 = 1.0，但无校验。

---

### 3.2 `hook.py` — VisualTokenPruningHook

**规模**: ~641 行，MRO 继承链：`VisualTokenPruningHook → HookFastRuntimeMixin → HookLegacyRuntimeMixin → HookDiagnosticsMixin → HookGeometryMixin`

**关键机制**:

#### `_compression_backend()` (hook.py:162-166)
```python
def _compression_backend(self) -> str:
    backend = getattr(self.config, "acgtp_compression_backend", "projector")
    if getattr(self.config, "acgtp_internal_pruning_enabled", False):
        backend = "internal"
    return "internal" if backend == "internal" else "projector"
```
注意：`acgtp_internal_pruning_enabled=True` 会强制覆盖 `acgtp_compression_backend`。但 `acgtp_compression_backend="internal"` 也会让 `_internal_pruning_requested()` 返回 True。

#### `_projector_hook` (hook.py:572-583)
```python
def _projector_hook(self, module, inputs, output):
    with torch.no_grad():
        pruned, metrics = self._run(output)  # ← 统一入口
    metrics.timing.hook_total_ms = ...
    self._latest_stats = metrics.to_fast_eval_stats()
    return pruned
```
- 无论 internal 还是 projector 后端，都走 `_run()` 入口。
- internal 后端时 `_run()` 调用 `_finalize_acgtp_fast_runtime`，后者调用 `_prepare_internal_pruning_plan` 而非实际剪枝 tensor。

#### 已知问题:
1. **`_internal_pruning_requested()` 在 `attach_to_model` 中被调用两次** (hook.py:512, 546):
   - 第 512 行: `if self._internal_pruning_requested()` — 检查是否需要 internal 后端
   - 第 546 行: `if language_model is not None and not self._internal_pruning_requested()` — 检查是否注册 LLM pre-hook
   - 这两处调用 `enable_acgtp_internal_pruning` 后，`self._internal_backend` 已设置，`_compression_backend()` 结果不变。但逻辑上 `enable_acgtp_internal_pruning` 可能失败并返回 None，此时第 532-543 行的错误处理会修改 `acgtp_compression_backend` 或抛出异常。

2. **internal 后端下 `_language_model_pre_hook` 不会被注册**（第 546 行条件排除），这是正确的——internal 后端不需要 position preserve。

3. **`reset_step()` 只清理 `self._post_pruning`**，但不清理 `_acgtp_latency_plan_cache`（需要 `_reset_latency_plan_cache()` 显式调用，在 `_mark_warmup_step` 中调用）。

---

### 3.3 `internal/backend.py` — ACGTPInternalPruningBackend

**规模**: ~1820 行

#### monkey-patch 机制 (backend.py:455-477)

```python
def attach(self):
    self.backbone._acgtp_internal_original_forward = self.backbone.forward
    self.backbone._acgtp_internal_backend = self
    wrapped_forward = _make_acgtp_internal_forward(
        self.backbone._acgtp_internal_original_forward, self)
    setattr(wrapped_forward, "_acgtp_internal_wrapped", True)
    self.backbone.forward = types.MethodType(wrapped_forward, self.backbone)
```

关键点：
- **使用 `types.MethodType`** — 这是正确的，因为 `wrapped_forward` 是闭包捕获了 `self`（backbone）的普通函数，不需要 `MethodType` 也能工作，但这里用 `MethodType` 显式绑定了 `self`。
- 保存 `original_forward` 和 `backend` 引用到 `backbone` 上，便于 detach 恢复。

#### `_make_acgtp_internal_forward` (backend.py:1380-1783)

完整的 monkey-patch forward，包含：
1. **Decode 路径** (backend.py:1400-1439): 跳过 monkey-patch 逻辑，直接调用 `original_forward`。仅记录 bookkeeping 和（可选的）decode attention capture。
2. **Prefill 路径** (backend.py:1441-1776): 手动执行 layers 循环，在 `prune_layer` 处调用 `resolve_visual_keep_indices`。

#### `resolve_visual_keep_indices` (backend.py:873-1374)

- 约 **500 行**，三种模式:
  1. **`latency_fast_path`** (backend.py:968-1080): 调用 `select_internal_quota_tokens` 的 fast 路径，返回纯 tensor 结果，无详细诊断。
  2. **`diagnostic_only`** (backend.py:1172-1238): 返回 fallback keep indices，填充诊断字段。
  3. **完整路径** (backend.py:1240-1374): 调用 `select_internal_quota_tokens`，填充全部 ~80 个诊断字段。

#### 两级缓存系统

| 缓存 | 位置 | 命中条件 | 用途 |
|------|------|----------|------|
| `latency_internal_keep_cache` | backend | hook 侧 latency_plan_cache hit + key 匹配 | 缓存 `visual_keep_indices` tensor |
| `latency_seq_tensor_cache` | backend | 在 `latency_internal_keep_cache` 内部 | 缓存 `keep_indices` / `position_ids` / `cache_position` tensor |

**潜在问题**:
1. **`batch_size > 1` 被静默跳过** (backend.py:678-686): seq_tensor_cache 不支持 batch > 1，直接返回 None。
2. **Decode 路径不执行 `resolve_visual_keep_indices`** — decode 时（`is_prefill=False`）直接调用 `original_forward`，不进行任何剪枝决策。这意味着 decode 阶段 KV cache 不会被剪枝（除非 prefill 阶段已剪枝）。
3. **`_current_visual_keep_tensor` 和 `_current_visual_keep_indices` 在每次 prefill 时更新**，但 decode attention capture 使用这些值，可能存在陈旧性。

---

### 3.4 `runtime/fast.py` — HookFastRuntimeMixin

**规模**: ~1230 行

#### `_run_acgtp_fast_runtime` (fast.py:647-1226)

完整执行流程：

```
1. 提取 robot gripper 位置
2. 检查 LatencyPlanCache hit → 如果命中直接返回缓存的 keep_indices
3. 从 TokenGeometryCache 采样 token depth
4. valid_depth_mask 计算
5. 静态场景缓存 lookup/store
6. depth_edge_scores
7. scene_layout_scores
8. contact_ring_scores
9. motion_corridor_scores
10. future_action_constraint_scores
11. ACGTPHistoryBuffer 更新（如果启用）
12. decide_acgtp_dynamic_budget（动态相位）
13. ACGTPAttentionGuide（仅 projector 模式，internal 模式禁用）
14. select_acgtp_v2_fast
15. _finalize_acgtp_fast_runtime
    → internal: _prepare_internal_pruning_plan
    → projector: index_select + _prepare_position_preserve_info
```

#### `_build_internal_geometry_payload` (fast.py:530-645)

将 hook 侧计算的几何分数打包成 dict，通过 `InternalPruningPlan.geometry_payload` 传递给 internal backend。

**构建的 payload**:
- `scene_scores`, `depth_scores`, `contact_scores`, `motion_scores`, `action_constraint_scores` (各 256 维)
- `geo_protect_mask`: 硬保护 mask（超过 quantile 的 contact/action/motion 证据）
- `layout_score`, `contact_score`, `motion_score`: 功能分支分数
- `valid_mask`, `constrained_fill_mask`
- `quota_config`: 来自 `build_internal_quota_config`
- `dynamic_decision`: ACGTP 动态相位决策

#### 潜在问题:
1. **`_build_internal_geometry_payload` 每次都完整构建**，即使 internal backend 不在 prefill 层做决策（decode 阶段 payload 可能未被使用，但仍然计算）。
2. **`select_acgtp_v2_fast` 的 `w_semantic=0.20` 是硬编码** (fast.py:1142)，不跟随配置。

---

### 3.5 `methods/functional_quota.py` — select_internal_quota_tokens

**规模**: ~480 行

两套实现：

| 路径 | 条件 | 特点 |
|------|------|------|
| `_select_internal_quota_tokens_fast` | `latency_fast_path=True` | 全 GPU tensor 操作，无 CPU 转换，无详细诊断 |
| `select_internal_quota_tokens` | 默认 | CPU/GPU 混合，含完整 branch 归属诊断 |

**分配算法** (functional_quota.py:35-52):
```python
def allocate_branch_quotas(total, weighted_names):
    # 1. 按权重比例分配（floor）
    # 2. 按余数从大到小分配（确保总和 = total）
```

**Selection 流程** (functional_quota.py:83-230):
```
1. explicit_protect_mask → geo 保护（contact/action/motion evidence 超过 quantile）
2. 若保护集 > target_k → 扩充 budget
3. 剩余 budget 按 quota 分配：layout / contact / motion / sem / hist
4. fill 填充（geo/sem/hist max-score）
5. fallback（原始 keep_indices 兜底）
6. 最终 fallback（全 valid tokens）
```

**潜在问题**:
1. **`branch_selected_sum` 可能 > `target_k`** — 因为 hard_protect + functional branches + fill + fallback 可能叠加。但 `final_mask` 取 unique，最终 `torch.unique()` 后去重。
2. **`fallback` 兜底可能覆盖其他分支** — 如果 `selected_count < target_k`，直接添加 fallback indices，不检查是否已在其他分支选中（虽然 `add_many` 内部有 `idx_i in selected` 检查）。

---

### 3.6 `signals/robot.py` — TokenGeometryCache & ACGTPStaticSceneCache

**规模**: ~1870 行

#### TokenGeometryCache

- 缓存 token → 3D 坐标映射（基于 camera intrinsics/extrinsics）
- key: `(depth.shape, camera_intrinsics_hash, T_robot_cam_hash)`
- `sample_depth(depth, cache)`: 根据 depth 值从预计算的 rays 中采样

#### ACGTPStaticSceneCache

- 缓存 `edge_scores` 和 `scene_result`
- 命中条件: depth delta < threshold AND valid_iou > threshold
- 节省: 约 **0.17 ms** (fast.py:847) 的 scene layout 计算

---

### 3.7 `core/metrics.py` — HookMetrics

**规模**: ~1480 行

`HookMetrics` dataclass 包含 **~600 个字段**（含大量别名和诊断字段）。两个序列化路径：

| 方法 | 用途 | 字段数 |
|------|------|--------|
| `to_fast_eval_stats()` | ACGTP fast runtime（推理时） | ~200 个关键字段 |
| `to_eval_stats()` | 完整诊断 | 所有字段 |

**关键统计**:
- **pruning_result**: dict，包含 `num_tokens_before/after`、`actual_keep_ratio`、`compression_backend`、`projector_pruning_applied`
- **compression_backend**: `"projector"` 或 `"internal"`
- **timing**: `HookTiming` 子 dataclass

**潜在问题**:
1. **`to_fast_eval_stats()` 有 ~300 行逻辑**，用于 alias 解析和字段填充。逻辑复杂，容易出现遗漏字段。
2. **大量字段存在但从未被写入**（如 `_qk_text_to_visual_attention` 相关字段）。

---

### 3.8 `runtime/post.py` — PostPruningStateManager

**规模**: ~227 行

两个核心职责：

#### Position Preserve（projector 后端）
- `prepare_position_preserve_info()`: 将 `keep_indices` 存入 `_pending_position_info`
- `language_model_pre_hook()`: 
  - Prefill: 注入 `position_ids`，保留原始 RoPE positions
  - Decode: 追加 decode tokens 的 position_ids（从 prefill 最后一个 position 继续）

#### Internal Pruning Plan（internal 后端）
- `prepare_internal_pruning_plan()`: 创建 `InternalPruningPlan` 并调用 `backend.set_pending_plan()`

**潜在问题**:
1. **`language_model_pre_hook` 中对 `kwargs["position_ids"]` 的设置在 `prefill` 时可能与 HuggingFace 自动计算的 position_ids 冲突**。第 208-209 行检查 `if kwargs.get("position_ids") is not None: return`，但 prefill 时 HF 不会自动计算（`inputs_embeds` 路径通常 position_ids=None）。

---

## 四、完整数据流图

```
[INPUT]
    │
    ▼
[Vision Encoder (ViT)]
    shape: [B, 256, D_vision]
    │
    ▼
[Projector (MLP)]
    shape: [B, 256, D_hidden]  (D_hidden=4096 for LLaMA-7B)
    │
    ▼
[register_forward_hook: _projector_hook]
    │
    ├─► geometry_recorder.get_latest()          (提取 depth, rgb, camera)
    ├─► TokenGeometryCache.sample_depth()        (~0.04ms)
    ├─► compute_depth_edge_scores()              (~0.17ms)
    ├─► compute_scene_layout_scores()
    ├─► compute_contact_ring_scores()
    ├─► compute_motion_corridor_scores()
    ├─► compute_future_action_constraint_scores()
    ├─► select_acgtp_v2_fast()                   (返回 keep_indices)
    │
    ├─► [if backend=projector]
    │       visual_tokens = index_select(keep)
    │       PostPruningStateManager.prepare_position_preserve_info()
    │       ↓
    │   [register_forward_pre_hook: _language_model_pre_hook]
    │       Injects preserved position_ids for prefill
    │
    └─► [if backend=internal]
            PostPruningStateManager.prepare_internal_pruning_plan()
                → InternalPruningPlan(keep_indices, geometry_payload, quota_config)
                → backend.set_pending_plan()
            visual_tokens = UNCHANGED
    │
    ▼
[LLM Input: [B, visual_K + text_tokens, D_hidden]]
    │
    ├─► [Prefill Path]
    │       If internal_backend:
    │           manual layer loop in _make_acgtp_internal_forward
    │               layer 0: full forward
    │               layer 1: full forward
    │               layer 2: resolve_visual_keep_indices() ← re-evaluate with LLM attention
    │                         + hidden_states = index_select(keep)
    │                         + position_ids = keep_indices
    │                         + cache_position = keep_indices
    │               layers 3-31: forward(shortened)
    │           Else (projector):
    │               standard LLM forward + position_ids injected by pre-hook
    │
    └─► [Decode Path]
            If internal_backend:
                bypass monkey-patch → original_forward
            Else (projector):
                original_forward with preserved position_ids
```

---

## 五、发现的问题汇总

### 5.1 配置与默认值问题

| 编号 | 文件 | 位置 | 问题 | 严重度 |
|------|------|------|------|--------|
| C1 | config.py | 260-267 | `acgtp_internal_*_quota_ratio` 6 个字段总和应=1.0，但无校验 | 低 |
| C2 | config.py | ~280 | 280 个字段，40+ 从未在当前路径使用 | 低（维护性） |
| C3 | config.py | 227-228 | `acgtp_internal_pruning_enabled` 和 `acgtp_compression_backend="internal"` 存在冗余映射关系 | 低 |
| C4 | config.py | 293 | `acgtp_runtime_mode` 的 `from_eval_cfg` 校验只接受 `fast/debug/audit`，但默认值是 `"fast"` | 无问题 |

### 5.2 数据一致性问题

| 编号 | 文件 | 位置 | 问题 | 严重度 |
|------|------|------|------|--------|
| D1 | fast.py | 1100-1113 | internal 模式下 `_build_acgtp_attention_guide` 被跳过（attention_guide 全置 False），但 `geometry_payload` 仍然传入 `quota_config`。`quota_config` 中的 `attention_enabled` 在 `resolve_visual_keep_indices` 中被读取（第 901 行）。 | 低（行为正确但绕路） |
| D2 | fast.py | 558 | `_build_internal_geometry_payload` 中的 `build_internal_quota_config` 传入 `hard_ratio`，但 `resolve_visual_keep_indices` 内部也重新计算 `hard_ratio`。**两处都算了一次，但只有后者生效**。 | 低（冗余计算） |
| D3 | fast.py | 70 | `_finalize_acgtp_fast_runtime` 中 `idx = idx[(idx >= 0) & (idx < int(num_tokens))]` 过滤掉越界索引，但 `select_acgtp_v2_fast` 返回的 `keep_indices_np` 应该已经是有效范围。这行是防御性代码。 | 无问题 |
| D4 | backend.py | 1027 | `resolve_visual_keep_indices` 返回的 `keep` 是 `torch.unique(keep, sorted=True)`，在 `seq_keep_indices_from_visual` 中会再次调用 `torch.unique`。**双重 unique**。 | 低（功能正确但冗余） |

### 5.3 潜在 Bug

| 编号 | 文件 | 位置 | 问题 | 严重度 |
|------|------|------|------|--------|
| B1 | post.py | 206-209 | `language_model_pre_hook` 在 `state is None` 时直接 return。如果 `position_preserve_enabled=True` 但第一次 prefill 前 `_pending_position_info` 被意外清空，后续 prefill 不会注入 position_ids，但也不会报错。 | 中 |
| B2 | fast.py | 725 | `depth_source_key = getattr(latest, "depth_metadata", {}) and latest.depth_metadata.get("source_key")` — 如果 `depth_metadata` 是空 dict `{}`，则 `and` 短路返回 `{}`（truthy），但不是 None。然后传给 `_store_latency_plan_cache` → `depth_source_key` = `{}`。后续 `cached_plan.get("depth_source_key")` 返回 dict 而非 string，但 `to_fast_eval_stats` 第 1145 行只是读取不处理。 | 低（数据污染） |
| B3 | backend.py | 1466 | `batch_size != 1` 时 raise RuntimeError，但 `disable_acgtp_internal_pruning` 后的后续 forward 仍然尝试使用 `original_forward`，这是正确的。但 `last_info` 中记录了 `"unsupported_batch_size"`，若下次 prefill batch=1 则会恢复。**无清理机制**。 | 低 |
| B4 | fast.py | 1216 | `depth_source_key` 被存储到 latency cache（第 500 行），但 `_lookup_latency_plan_cache` 中（第 440 行）检查 `depth_probe` 时未检查 `depth_source_key` 变化。若图像源改变（如 camera 切换），depth 值可能不变但语义已变，cache 仍会命中。 | 中（隐蔽） |

### 5.4 性能问题

| 编号 | 文件 | 位置 | 问题 | 严重度 |
|------|------|------|------|--------|
| P1 | backend.py | 1400-1439 | **Decode 路径不触发 internal pruning**。Decode 时（`is_prefill=False`）直接调用 `original_forward`，KV cache 完全不受剪枝影响（除非 prefill 阶段已剪枝）。这是架构决定，但意味着剪枝加速主要来自 prefill 阶段的序列缩短，decode 阶段无直接加速。 | 架构限制 |
| P2 | fast.py | 530-645 | **`_build_internal_geometry_payload` 在每步都执行**，即使 internal backend 在 decode 阶段不触发。对于 `latency_fast_path`，geometry 分数来自 `geometry_payload`（已在 hook 侧计算），但 `build_internal_quota_config` 每次都重新构建 ~50 字段的 dict。 | 低（内存分配开销） |
| P3 | backend.py | 1565-1571 | `capture_for_pruning` 条件检查在每层都执行，但只在 `prune_layer` 层有意义。其他层 `capture_for_pruning=False`，但 `try_materialized_attn_requested` 可能导致不必要的 attention computation。 | 低 |
| P4 | functional_quota.py | 298 | `add_many` 中 `for raw_idx in indices.detach().cpu().tolist()` — 完整的非 fast 路径将 GPU tensor 转到 CPU 进行 Python 循环。**这是最大性能瓶颈**。`select_internal_quota_tokens`（非 fast 路径）约比 fast 路径慢 10x。 | 高 |

### 5.5 Metric/Instrumentation 问题

| 编号 | 文件 | 位置 | 问题 | 严重度 |
|------|------|------|------|--------|
| M1 | eval_openvla_baseline.py | summary.json | `mean_llm_forward_time_ms` / `mean_vision_encoder_time_ms` / `mean_projector_time_ms` 均为 **null**。无法知道 244.4ms model forward 具体花在哪里。 | 高（无法诊断） |
| M2 | metrics.py | ~1100-1474 | `to_fast_eval_stats()` 约 370 行，`to_eval_stats()` 约 110 行。大量 alias 映射容易遗漏新字段。 | 低（维护性） |
| M3 | eval_openvla_baseline.py | 244-250 | `_build_runtime_fingerprint` 返回 `mean_llm_prefill_time_ms: null` / `mean_llm_decode_time_ms: null`。没有 prefill/decode 计时拆分。 | 高（无法诊断） |

---

## 六、Critical 问题优先级

### 🔴 必须修复（影响正确性）

| 编号 | 问题 | 影响 |
|------|------|------|
| M1 | `mean_llm_forward_time_ms` 等 null | 无法诊断 244ms 瓶颈，无法判断剪枝是否有效 |
| B1 | `position_preserve` 静默失效 | projector 后端在边界情况下可能不注入 position_ids，导致 RoPE 错位 |
| B4 | `depth_source_key` cache 污染 | camera 切换时缓存不感知，可能返回错误的 keep 决策 |

### 🟡 建议修复（影响性能）

| 编号 | 问题 | 影响 |
|------|------|------|
| P1 | Decode 路径不触发 internal pruning | 剪枝在 decode 阶段无效，KV cache 不缩短 |
| P4 | 非 fast 路径 CPU/GPU 混合导致 10x 慢 | 调试/诊断模式下性能下降 |
| D2 | `_build_internal_geometry_payload` 冗余调用 | minor overhead |

### 🟢 可改进（维护性）

| 编号 | 问题 | 影响 |
|------|------|------|
| C2 | 280 个配置字段 | 难以维护，大量字段无作用 |
| C1 | quota ratio 无总和校验 | 可能配置出无效 quota 分配 |
| P2 | geometry payload 每步重建 | minor memory/CPU overhead |

---

## 七、测试覆盖

| 测试文件 | 测试内容 | 覆盖情况 |
|----------|----------|----------|
| `behavior_regression.py` | 行为回归：hook attach/detach、metric 字段完整性、quota 计算正确性 | 基础路径覆盖 |
| `validation_tests.py` | 验证测试：temporal/fallback/keep_ratio/phase accounting | 较全面 |

**缺失的测试**:
- Decode 路径正确性（`is_prefill=False` 绕过 monkey-patch）
- `position_preserve` 的 prefill/decode 边界情况
- `depth_source_key` 变化时的 cache invalidation
- `batch_size > 1` 时的错误处理
- Internal backend + projector fallback 组合

---

## 八、当前主线配置（functional_quota_static_050_layer2）

```yaml
pruning_strategy: robot_geo_acgtp_v2
keep_ratio: 0.50
compression_backend: internal
internal_pruning_enabled: true
internal_prune_layer: 2
acgtp_fast_selector_enabled: true
acgtp_dynamic_enabled: true
acgtp_static_scene_cache_enabled: true
acgtp_latency_plan_cache_enabled: false
acgtp_history_enabled: false
acgtp_attention_guidance_enabled: false
acgtp_internal_functional_quota_enabled: true
acgtp_internal_selection_mode: geo_guarded
acgtp_internal_attention_enabled: true
acgtp_internal_latency_fast_path: false
```

---

## 九、结论

当前代码在 `functional_quota_static_050_layer2` 配置下**功能正确**，但存在三个关键障碍：

1. **无法诊断瓶颈**：`mean_llm_forward_time_ms` 等为 null，244ms 的耗时分布未知。
2. **Decode 阶段无剪枝效果**：这是架构决定，不是 bug，但限制了加速上限。
3. **Hidden 问题**：`depth_source_key` cache 污染、`position_preserve` 边界情况等在当前测试场景下可能未触发。

**下一步建议**：
1. 补充 LLM/ViT/Projector 分步计时（最高优先）
2. 确认 decode KV cache 是否真的被 prefill 剪枝缩短
3. 修复 B4 (`depth_source_key` cache)
4. 确认 functional quota 分配是否按预期工作
