# 评测指标对齐修改报告 (2026-05-30)

本次修改把 `scripts/build_performance_report.py` 的报告指标对齐到五篇 VLA 高效推理
工作(VLA-ADP / VLA-IAP / VLA-Pruner / VLA-Cache / DepthCache,见
`docs/vla_efficient_inference_survey_20260530.md`)的评测惯例:同时报告
**保留率 / 成功率 / 端到端加速 / 机制证据** 四类指标,而不是用单一的
`cuda_speedup_vs_baseline` 当唯一裁判。

> 约束:本次**只改报告/统计层**。核心剪枝逻辑
> (`pruning/hook.py`、`pruning/internal_pruning.py`、`pruning/selector.py` 的
> token 选择、剪枝位置、forward)**一行未动**。

## 0. 一句话动机

base OpenVLA 是自回归 + decode 主导(decode ≈ 80% LLM 时间),端到端
wall/cuda speedup 在 base 上有物理天花板。继续用它当主裁判,会把
"选得准、机制成立、只是被 decode 地板挡住"的方法**误判为失败**。对齐后,
报告把端到端 speedup 降级为辅助指标,把 **prefill 时间 + 序列/FLOP 缩减**
提升为机制主指标。

## 1. 修改的文件

| 文件 | 改动 |
|---|---|
| `scripts/build_performance_report.py` | 新增派生指标、分层 verdict、三个 markdown 小节、扩展 CSV 列 |

核心剪枝文件时间戳确认未变(hook.py / internal_pruning.py / selector.py 均为本次修改之前的时间)。

## 2. 新增指标与计算公式

### 2.1 prefill / decode 速度(扩展原 speedup 循环)
| 指标 | 公式 |
|---|---|
| `prefill_speedup_vs_baseline` | `baseline.lm_prefill_ms / row.lm_prefill_ms` |
| `prefill_saved_ms_vs_baseline` | `baseline.lm_prefill_ms − row.lm_prefill_ms` |
| `decode_speedup_vs_baseline` | `baseline.lm_decode_observed_ms / row.lm_decode_observed_ms` |
| `decode_saved_ms_vs_baseline` | `baseline.lm_decode_observed_ms − row.lm_decode_observed_ms` |
| `prefill_share_of_llm` | `lm_prefill_ms / llm_ms` |
| `decode_share_of_llm` | `lm_decode_observed_ms / llm_ms` |

### 2.2 序列 / FLOP 机制证据
| 指标 | 公式 |
|---|---|
| `internal_seq_retention` | `internal_seq_kept_mean / internal_seq_original_mean` |
| `internal_seq_reduction_pct` | `1 − internal_seq_retention` |
| `theoretical_attention_flop_ratio` | `seq_retention²` |
| `theoretical_attention_speedup` | `1 / theoretical_attention_flop_ratio` |
| `theoretical_linear_flop_ratio` | `seq_retention` |
| `theoretical_linear_speedup` | `1 / theoretical_linear_flop_ratio` |

### 2.3 internal-aware 诚实 FLOP(对 GPT 提示词的关键校正)
naive 的 `seq_retention²` 假设**全部层**都跑缩短序列。但 internal backend 在
layer K **之后**才剪,layers 0..K 仍跑全长,所以 naive 值对 internal 是**高估**。
新增按"受益层占比"加权的诚实估计:

| 指标 | 公式 |
|---|---|
| `internal_benefiting_layer_fraction` | `(L − 1 − K) / L`(K=prune_layer,L=总层数;非 internal=1.0) |
| `internal_effective_attention_speedup` | `1 / [benefit·seq² + (1−benefit)·1]` |
| `internal_effective_linear_speedup` | `1 / [benefit·seq + (1−benefit)·1]` |

报告把 naive 列标注为"上界 UB(适用于 projector / pre-layer-0 剪枝)",
把 internal-eff 列作为对比 wall 的诚实值。
实测:`internal_geo_guarded_050` seq retention 0.556 → 理论 **3.23x(UB)**,
诚实 **2.67x**(benefit_frac 0.906,prune_layer=2,L=32)。

### 2.4 theory-vs-wall 缺口
| 指标 | 公式 |
|---|---|
| `wall_gain_vs_prefill_gain_ratio` | `wall_saved / prefill_saved` |
| `llm_gain_vs_prefill_gain_ratio` | `llm_saved / prefill_saved` |

所有除法走 `_safe_div`(None / 0 / NaN 返回 None,不抛错)。

## 3. 分层 verdict(端到端降级为辅助)

不再用 `cuda_speedup < 1.02 → WEAK` 一票否决。新判定顺序:

1. SR 掉点 > 阈值(`max(5%, 1/episodes)`)→ `QUALITY_REGRESSION`
2. `wall_speedup > 1.02` → `END_TO_END_SPEEDUP`
3. prefill **未测量**(N/A)→ `QUALITY_OK_MECHANISM_UNMEASURED`
   / `GEO_GUARD_BREACH_MECHANISM_UNMEASURED`
   (防假阴性:不能用没测的指标判方法死刑)
4. `prefill_speedup ≤ 1.0` → `NO_PREFILL_GAIN`
5. prefill 增益 + `geo_critical == 0` → `MECHANISM_VALID_BUT_END_TO_END_DECODE_BOUND`
6. prefill 增益但 guard 失守 → `MECHANISM_PARTIAL`

原 verdict 保留为 `legacy_verdict` 字段(向下兼容)。

## 4. markdown 新增三节

- `## Paper-aligned Metrics` — 四类指标对齐说明 + 总表(每方法一行)
- `## Prefill-vs-Decode Breakdown` — prefill/decode 拆分 + decode-bound 判定
- `## Theory-vs-Wall Gap` — 理论(UB)vs 诚实(internal-eff)vs wall 三列对比

原有所有节、字段、CSV 列全部保留;新列**追加**在 CSV 末尾。

## 5. 为什么更对齐五篇论文

五篇都同报四类指标,且**没有一篇**用端到端 speedup 单点否决方法。
VLA-Pruner 更主动报"FLOP 理论 ~3.3x vs wall 实测 ~1.83x"的缺口——本次新增的
theory-vs-wall 节正是复现这一诚实叙事。改完后:
(a) prefill 时间 + 保留率 + 序列缩减升为主指标;
(b) 端到端 wall 在 base 上明确标为辅助;
(c) verdict 把质量 / 机制 / 端到端三件事分开,不再混为一谈。

## 6. 是否修改了核心剪枝逻辑

**没有。** token 选择、prune_layer、forward 全部原样。仅报告/统计层改动。

## 7. 最小验证(已完成)

- `python -m py_compile scripts/build_performance_report.py` → OK
- 实跑 `outputs/core_surface_3task3trial`(含 4 个核心方法)→ json/md/csv 三件全生成。
- 结果(本数据用 `eval_openvla_baseline.py` 跑,**未拆 prefill/decode**,故 prefill 列正确显示 N/A,未编造):

| 方法 | SR | seq retain | 理论 attn(UB) | internal-eff | wall | geo_critical | verdict |
|---|---|---|---|---|---|---|---|
| baseline_none | 8/9 | — | — | — | 1.000x | — | BASELINE |
| projector_acgtp_legacy_050 | 6/9 | — | — | — | 1.035x | — | QUALITY_REGRESSION |
| internal_geo_guarded_050 | 7/9 | 0.556 | 3.23x | 2.67x | 0.967x | 0 | QUALITY_OK_MECHANISM_UNMEASURED |
| internal_dynamic_050 | 7/9 | 0.842 | 1.41x | 1.36x | 0.961x | 0 | QUALITY_OK_MECHANISM_UNMEASURED |

## 8. 后续:retention sweep(对齐 retention-vs-speedup-vs-SR 曲线)

要让 prefill 列填上真实值,需用会挂 LM-call 计时 hook、产出
`lm_prefill_time_ms_observed` 的 `scripts/probe_pruning_compute_reality.py`
(`eval_openvla_baseline.py` 的 CUDATimer 覆盖整个 predict_action 但不拆 prefill/decode)。

对 ρ = 1.0 / 0.65 / 0.50 / 0.35 各跑一个评测目录,再聚合成曲线:

```bash
for rho in 1.0 0.65 0.50 0.35; do
  python scripts/eval_openvla_baseline.py \
    --pretrained_checkpoint /infini-data/checkpoints/openvla-7b-finetuned-libero-spatial \
    --acgtp_compression_backend internal \
    --acgtp_internal_pruning_enabled true \
    --acgtp_internal_selection_mode geo_guarded \
    --keep_ratio ${rho} \
    --run_root_dir outputs/retention_sweep/geo_guarded_${rho}
done

python scripts/build_performance_report.py \
  --root outputs/retention_sweep --baseline geo_guarded_1.0 \
  --prefix retention_sweep_report
```

预期:`internal_seq_retention` 随 ρ 单调下降、`theoretical_*_speedup` 上升;
若 wall 不随之上升,theory-vs-wall 节会量化暴露 decode 地板 / hook overhead,
而非 token 选择本身的问题。这与设计宪法的方向 A 一致:base 只证
"选择质量 + 机制有效",端到端加速兑现留给 prefill 主导的 OFT/pi0。
