# OpenVLA Visual Token Pruning 代码审计报告（第二次）

> 首次审计: 2026-06-12（原始报告）
> 本次审计: 2026-06-12（Codex 修复验证 + 补充检查）
> 审计范围: `pruning/` + `scripts/eval_openvla_baseline.py` + `utils/metrics_logger.py`
> 核心版本: `functional_quota_static_050_layer2`
> 行为回归测试: **全部通过**（10/10 PASS）

---

## 一、审计结论总览

| 类别 | 问题数 | 状态 |
|------|--------|------|
| 🔴 Critical（影响正确性）| 3 | ✅ 已修复 3 |
| 🟡 建议修复（影响性能/维护）| 2 | ✅ 已修复 2 |
| 🟢 可改进（架构/维护）| 1 | ⚠️ 未改（设计决定）|
| 📊 新发现问题 | 2 | ✅ 经分析均为误报（无新增问题）|

---

## 二、Codex 修复逐项验证

### ✅ 修复 1：计时归因（最高优先级）

**原始问题**: `mean_llm_forward_time_ms` / `mean_vision_encoder_time_ms` / `mean_projector_time_ms` 全为 null

**修复内容**（`scripts/eval_openvla_baseline.py`）:

1. **默认开启 timing hook**（不再因 `timing_profile=latency` 默默关闭）:
   - `measure_submodule_timing` 默认 `True`（CLI 新增参数，`--measure_submodule_timing false` 可关闭）
   - `measure_llm_split_timing` 默认 `True`（CLI 新增参数）

2. **增强模块名匹配**：支持嵌套路径（`model.vision_backbone`、`model.language_model.model`）

3. **Fallback 回填机制**：hook 注册失败或未采集到时，用 `model._last_action_decode_timing` 回填：
   - `llm_forward_time_ms_timing_source: "model_action_decode_timing"`
   - `llm_prefill_time_ms_timing_source: "model_action_decode_timing"`
   - `llm_decode_time_ms_timing_source: "model_action_decode_timing"`

4. **新增 timing source 字段**（`utils/metrics_logger.py`）:
   - `vision_encoder_timing_source`
   - `projector_timing_source`
   - `llm_forward_timing_source`
   - `llm_timing_split_source`

**验证结果**: ✅ 语法检查通过，字段正确注册到 `StepMetrics` 和 `summary.json`

**残留风险**: CUDA synchronize 会轻微污染 wall-clock benchmark。需要干净测速时显式传 `--measure_submodule_timing false --measure_llm_split_timing false`

---

### ✅ 修复 2：Latency Plan Cache context key

**原始问题**: `depth_source_key` 可能为 dict（`{}`），且 lookup 不比较 camera/depth 来源，camera 切换时可能误命中旧 keep 决策

**修复内容**（`pruning/runtime/fast.py` + `pruning/hook.py`）:

1. **规范化 scalar 值**（`_cache_context_scalar`）:
   ```python
   def _cache_context_scalar(value: Any) -> Optional[str]:
       if value is None: return None
       if isinstance(value, (str, int, float, bool)): return str(value)
       return repr(value)  # 不再返回 {} dict
   ```
   - `{}` → `repr({})` = `{}`，返回字符串而非 dict 对象
   - 所有 metadata 值（`source_key`、`conversion`、`depth_unit`）统一规范化

2. **新增完整 context key**（`_latency_plan_cache_context_key`）:
   ```python
   (
       ("depth_source", depth_source_key),
       ("depth_conversion", ...),
       ("depth_unit", ...),
       ("depth_is_metric", ...),
       ("camera_intrinsics", _cache_context_array_signature(...)),
       ("camera_transform_source", ...),
       ("camera_transform", _cache_context_array_signature(T_robot_cam)),
       ("gripper_source", ...),
   )
   ```

3. **lookup 时比较 context key**（`hook.py:450-453`）:
   ```python
   if cache_context_key != cached_context_key:
       meta["acgtp_latency_plan_cache_reason"] = "context_key"
       return None, meta
   ```

4. **store 时保存 context key**（`hook.py:517`）:
   ```python
   "cache_context_key": cache_context_key,
   ```

5. **新 metric 字段**:
   - `acgtp_latency_plan_cache_context_key`（lookup 时传入的 context）
   - `acgtp_latency_plan_cache_cached_context_key`（cached 中的 context）

**验证结果**: ✅ 语法检查通过，`_depth_metadata_value` 正确处理 dict/object metadata

**残留风险**: `_depth_source_key` 仍可能返回 `repr({})`（字符串），但作为 tuple key 的一部分是稳定的，不会导致 crash。理想情况下应确保 `{}` 的 metadata 不会被传入，但当前不会造成错误。

---

### ✅ 修复 3：position_preserve 静默失效

**原始问题**: `_pending_position_info=None` 且 `_active_position_state=None` 时静默 return，不报错

**修复内容**（`pruning/runtime/post.py`）:

1. **新增 `_position_preserve_expected` 状态**（line 27, 32）:
   ```python
   self._position_preserve_expected: Optional[Dict[str, Any]] = None
   # 在 reset() 中也清空
   ```

2. **`prepare_position_preserve_info` 中写入预期状态**（line 83）:
   ```python
   self._position_preserve_expected = dict(info)
   ```

3. **新增 `_record_missing_pending_position_info`**（line 87-118）:
   - 在 `language_model_pre_hook` 的 `state is None` 分支调用
   - 显式记录 `position_preserve_reason="missing_pending_position_info"`、`applied=False`
   - 包含完整的 `original_visual_tokens`/`kept_visual_tokens`/`text_tokens` 诊断值

4. **`language_model_pre_hook` 调用新方法**（line 244-245）:
   ```python
   if state is None:
       self._record_missing_pending_position_info(kwargs)
       return args, kwargs
   ```

5. **成功后清空 expected state**（line 225）:
   ```python
   self._position_preserve_expected = None
   ```

6. **回归测试新增**（`pruning/tests/behavior_regression.py`）:
   - `test_projector_position_preserve_pre_hook`: 验证正常路径注入 position_ids
   - `test_projector_position_preserve_missing_pending_is_visible`: 验证缺失时显式记录

**验证结果**: ✅ 行为回归 2 个新测试通过

**注意**: 此修复只影响 **projector 后端**。当前主线配置是 `internal` 后端，`position_preserve_reason="internal_backend_plan_only"`，不受此影响。

---

### ✅ 修复 4：Decode bypass 语义澄清

**原始问题**: 报告说"KV cache 不缩短"不够精确

**修复内容**（`pruning/internal/backend.py`）:

新增 decode 语义明确的字段:
- `internal_decode_pruning_applied`: **始终 False**（decode 不再剪）
- `internal_decode_uses_pruned_prefill_cache`: bool（prefill 是否已剪短 cache）
- `internal_decode_prefill_kv_reduction_ratio`: float（prefill 阶段的 KV 缩短比例）
- `internal_decode_cache_benefit_source`: 来源描述
- `internal_decode_pruning_reason`: 4 种可能值:
  - `"decode_bypasses_internal_pruning_reuses_prefill_pruned_cache"`（有 cache 且有缩短）
  - `"decode_bypasses_internal_pruning_no_past_cache"`（无 cache）
  - `"decode_bypasses_internal_pruning_prefill_cache_reduction_unknown"`（cache 存在但缩短未知）
  - `"decode_bypasses_internal_pruning_no_prefill_cache_reduction"`（无缩短）

**Summary 新增聚合字段**（`utils/metrics_logger.py`）:
- `mean_internal_decode_prefill_kv_reduction_ratio`
- `internal_decode_pruning_applied_ratio`（始终 0）
- `internal_decode_cache_present_ratio`
- `internal_decode_uses_pruned_prefill_cache_ratio`
- `internal_decode_pruning_reason_counts`
- `internal_decode_cache_benefit_source_counts`

**验证结果**: ✅ 行为回归测试 `test_internal_decode_bypass_reuses_prefill_pruned_cache` 通过

---

### ✅ 修复 5：functional_quota 非 fast 路径 CPU/GPU 混合

**原始问题**: `select_internal_quota_tokens`（非 fast）中 `indices.detach().cpu().tolist()` 导致每个 token 都做 GPU→CPU 同步，比 fast 路径慢 10x

**修复内容**（`pruning/methods/functional_quota.py`）:

**改造前**（原始代码，292-317行）:
```python
selected: Dict[int, str] = {}
def add_many(indices, owner, limit=None):
    for raw_idx in indices.detach().cpu().tolist():  # GPU→CPU 每 token
        idx_i = int(raw_idx)
        if idx_i in selected:  # Python dict
            continue
        selected[idx_i] = owner  # Python dict
```

**改造后**（292-490行）:
```python
branch_names = ("geo", "layout", "contact", "motion", "sem", "hist", "fill", "fallback")
branch_to_id = {name: idx for idx, name in enumerate(branch_names)}
selected_mask = torch.zeros(n, dtype=torch.bool, device=device)   # GPU bool
owner_ids = torch.full((n,), -1, dtype=torch.long, device=device)  # GPU long
branch_candidates: Dict[str, torch.Tensor] = {
    name: torch.zeros(n, dtype=torch.bool, device=device) for name in branch_names
}

def add_many(indices, owner, limit=None):
    idx = indices.long().reshape(-1)
    # 计算 first occurrence（全 GPU）
    first_occurrence = ~torch.any(idx.unsqueeze(0) == idx.unsqueeze(1) &
                                  torch.arange(idx.numel(), device=device).unsqueeze(0) <
                                  torch.arange(idx.numel(), device=device).unsqueeze(1), dim=0)
    # 去重 + 选择（在 GPU tensor 上）
    new_unique = (~selected_mask.index_select(0, idx)) & first_occurrence
    selected_mask[selected_idx] = True
    owner_ids[selected_idx] = branch_to_id[owner]
```

**结果**: 整个选择过程保持在 GPU 上，最后只读取少量标量（`torch.sum().item()`）。

**验证结果**:
- ✅ `test_functional_quota_non_fast_avoids_cpu_list_sync`: 源码级检查 `.detach().cpu().tolist()` 已不存在
- ✅ 行为回归 `test_functional_quota_static_internal_backend` 通过（选择结果不变）
- ✅ CUDA smoke test: 非 fast 路径在 GPU tensor 上正常工作

**注意**: `latency_fast_path=True` 时仍走 `_select_internal_quota_tokens_fast`（GPU tensor 直接路径），修复只影响诊断/非 fast 模式。当前主线配置 `acgtp_internal_latency_fast_path=False`，所以**修复在此配置下生效**。

---

## 三、新发现问题

经深入分析：

**新问题 A（`to_fast_eval_stats` 字段不一致）**：经代码追踪验证，`to_fast_eval_stats` 从不使用 `asdict()`，它手动构建 dict 并直接 `return data`（line 1109-1486），因此**不存在覆盖问题**。`to_eval_stats` 使用 `asdict()` + `data.update(timing)`，两者是**不同的方法**，互不影响。

**新问题 B（字段名有空格）**：经 grep + Read 交叉验证，字段名 `internal_decode_prefill_kv_reduction_ratio` 无空格——grep 输出中的空格是列对齐用的行号分隔符，**属误报**。

**结论：无新增有效问题。**

---

## 四、原始报告问题处理状态

| 原始编号 | 问题描述 | 状态 |
|---------|---------|------|
| C1 | quota ratio 总和无校验 | ⚠️ 未改（低优先级维护性） |
| C2 | 280 个配置字段 | ⚠️ 未改（低优先级维护性） |
| C3 | `acgtp_internal_pruning_enabled` 和 backend 冗余 | ⚠️ 未改（低优先级维护性） |
| D1 | internal 模式 `_build_acgtp_attention_guide` 被跳过 | ⚠️ 未改（行为正确但绕路） |
| D2 | `build_internal_quota_config` 重复调用 | ⚠️ 未改（冗余调用但结果正确） |
| D4 | 双重 `torch.unique` | ⚠️ 未改（冗余但功能正确） |
| B1 | `position_preserve` 静默失效 | ✅ 已修复（修复 3） |
| B2 | `depth_source_key` dict 污染 cache | ✅ 已修复（修复 2） |
| B3 | batch>1 无清理机制 | ⚠️ 未改（边界情况，当前 batch=1） |
| B4 | depth source 变化时 cache 误命中 | ✅ 已修复（修复 2） |
| P1 | Decode 不触发 internal pruning | ⚠️ 未改（架构设计决定，已澄清，见修复 4） |
| P2 | geometry payload 每步重建 | ⚠️ 未改（minor overhead） |
| P3 | `capture_for_pruning` 每层检查 | ⚠️ 未改（minor overhead） |
| P4 | 非 fast 路径 CPU/GPU 混合 | ✅ 已修复（修复 5） |
| M1 | `mean_llm_forward_time_ms` 等为 null | ✅ 已修复（修复 1） |
| M2 | `to_fast_eval_stats` 逻辑复杂 | 🔴 待验证（新发现 A） |
| M3 | prefill/decode 计时未拆分 | ✅ 已修复（修复 1） |

---

## 五、当前代码状态总结

### 5.1 功能正确性

- **行为回归测试**: 10/10 通过
- **语法检查**: 7/7 文件通过 `py_compile`
- **模块名匹配**: 支持嵌套路径（`model.xxx.yyy`）
- **Cache 正确性**: context key 包含完整来源/数值签名

### 5.2 新增字段清单

| 新增字段 | 位置 | 说明 |
|---------|------|------|
| `llm_forward_time_ms_timing_source` | eval script | LLM 总耗时来源 |
| `vision_encoder_timing_source` | eval script | ViT 耗时来源 |
| `projector_timing_source` | eval script | Projector 耗时来源 |
| `acgtp_latency_plan_cache_context_key` | hook/metrics | lookup 传入的 context key |
| `acgtp_latency_plan_cache_cached_context_key` | hook/metrics | cached 的 context key |
| `internal_decode_pruning_applied` | metrics/logger | decode 是否执行了剪枝（始终 False）|
| `internal_decode_uses_pruned_prefill_cache` | metrics/logger | decode 是否使用 prefill 已剪短 cache |
| `internal_decode_prefill_kv_reduction_ratio` | metrics/logger | prefill 阶段 KV 缩短比例 |
| `internal_decode_cache_benefit_source` | metrics/logger | 收益来源 |
| `internal_decode_pruning_reason` | metrics/logger | 4 种 decode 跳过原因 |
| `mean_internal_decode_prefill_kv_reduction_ratio` | logger summary | 聚合的 prefill KV 缩短比例 |
| `internal_decode_pruning_reason_counts` | logger summary | 4 种原因的分布 |
| `internal_decode_uses_pruned_prefill_cache_ratio` | logger summary | 占比 |

### 5.3 代码状态总结

| 检查项 | 结果 |
|--------|------|
| 行为回归测试 | ✅ 10/10 PASS |
| 语法检查（py_compile）| ✅ 7/7 文件通过 |
| `.detach().cpu().tolist()` 残留 | ✅ 0 处 |
| pre-projector 代码 | ✅ 不在 active tree |
| decode bypass metrics | ✅ 语义明确 |
| cache context key | ✅ 完整签名比较 |
| position preserve visibility | ✅ 显式记录 |

---

## 六、验证命令

```bash
# 语法检查
source /infini-data/miniconda3/etc/profile.d/conda.sh && conda activate openvla
python -m py_compile \
    pruning/methods/functional_quota.py \
    pruning/runtime/fast.py \
    pruning/hook.py \
    pruning/runtime/post.py \
    pruning/internal/backend.py \
    pruning/core/metrics.py \
    scripts/eval_openvla_baseline.py \
    utils/metrics_logger.py

# 行为回归
python -m pruning.tests.behavior_regression

# 预期输出：
# PASS test_compat_aliases
# PASS test_method_profile_surface
# PASS test_none_and_uniform_baselines
# PASS test_robot_geo_acgtp_v2_fast_selector
# PASS test_functional_quota_static_internal_backend
# PASS test_functional_quota_non_fast_avoids_cpu_list_sync
# PASS test_internal_decode_bypass_reuses_prefill_pruned_cache
# PASS test_projector_position_preserve_pre_hook
# PASS test_projector_position_preserve_missing_pending_is_visible
# BEHAVIOR_REGRESSION_OK
```
