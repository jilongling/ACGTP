# Prefill-vs-保留率曲线 — internal geo_guarded (2026-05-30)

本文记录方向 A 三条收敛断言里 **断言②(机制有效:保留率能转成 prefill 计算下降)**
的定量验证结果。用 `scripts/probe_pruning_compute_reality.py`(会挂 LM-call 计时
hook、真实拆分 prefill/decode)在 base OpenVLA-7b 上对 internal geo_guarded 扫保留率。

> 只读验证,未改任何核心剪枝逻辑。probe 仅新增一个 `--geo_guarded_sweep` 驱动入口
> (附加 mode,不动 forward/选择/剪枝位置)。

## 1. 运行配置

- checkpoint:`/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial`(base OpenVLA,自回归)
- env:`openvla`(torch 2.2.0+cu121);任务 libero_spatial task_0;1 帧观测;warmup 2 + iters 8
- backend:internal,`prune_layer=2`,`selection_mode=geo_guarded`
- 保留率 ρ = 1.00 / 0.65 / 0.50 / 0.35
- 产物:`outputs/prefill_retention_sweep/`(json / csv / probe 自带 report / log)

## 2. 核心曲线

| ρ | seq kept/orig | visual kept | prefill ms | decode ms | hook ms | wall ms | geo_critical |
|---|---|---|---|---|---|---|---|
| 1.00 (后端基线) | 291/291 | 256 | 58.13 | 174.30 | 3.78 | 272.68 | 0 |
| 0.65 | 201/291 (0.691) | 166 | 42.93 | 177.14 | 3.68 | 260.48 | 0 |
| 0.50 | 163/291 (0.560) | 128 | 41.55 | 176.49 | 3.64 | 259.73 | 0 |
| 0.35 | 125/291 (0.430) | 90 | 42.59 | 181.98 | 3.93 | 267.76 | 0 |

参照:纯 baseline(`none@1.00`,无 hook、无后端)prefill = 44.87ms,decode = 177.88ms,wall = 260.58ms。
prefill 逐 iter 方差极小(sd < 0.7ms),曲线可信。

## 3. 结论

### 3.1 断言② 成立:机制有效
以"in-backend 参照"(ρ=1.00,剪枝已请求但保留全部 token,隔离掉后端/hook 固定开销)为分母:

| ρ | prefill 加速(vs in-backend 参照) |
|---|---|
| 0.65 | 1.354x |
| 0.50 | **1.399x** |
| 0.35 | 1.365x |

prefill 时间随保留率下降(58→42ms),**保留率确实转成了 prefill 计算的减少**。
这是方向 A 断言②要的定量证据:token 选择机制本身有效。

### 3.2 暴露的真实子发现:internal 后端有固定开销 + 拐点
- **后端固定开销 +13ms**:ρ=1.00 的 prefill 58.13ms vs 纯 baseline 44.87ms。
  来自 layers 0..2 跑全长 291 + 每次 prefill 的 index_select / causal_mask 重建 / DynamicCache。
  → 对**纯 baseline** 的净 prefill 加速只有 ~1.08x(41.55/44.87 的倒数附近),远小于"内部"1.40x。
- **0.35 拐点**:ρ 从 0.50 降到 0.35,prefill 不降反微升(41.55→42.59),decode 也升
  (176.49→181.98)。序列收益已被每步固定开销 + KV 跨层不一致开销吃掉。
  → **0.50 是当前 internal-at-layer-2 的最优 prefill 工作点。**

### 3.3 与端到端的关系(印证根因)
即便 prefill 在内部最多省 ~16ms,decode 始终 ~177ms(占 LLM ~80%)且完全不随 ρ 变。
wall 从未真正低于纯 baseline(最好 259.73 vs 260.58,基本持平)。
**这正是 theory-vs-wall 缺口:机制有效,但 base 自回归 decode 地板把端到端收益吃光。**
与 `acgtp_status_20260530.md` 的根因、`vla_efficient_inference_survey_20260530.md`
的"VLA-Pruner base 上 FLOP 3.3x→wall 1.83x"完全一致。

### 3.4 geo 硬保护全程守住
所有 ρ 下 `geo_critical=0`、`protected=52` 恒定。
断言①(几何硬保护不被误删)在保留率扫描下依然成立。

## 4. 收敛状态更新

| 断言 | 状态 |
|---|---|
| ① 选择质量(SR 守住 + geo_critical=0)| ✅ 达标(7/9,critical=0;本扫描再次确认 guard) |
| ② 机制有效(prefill 随 ρ 下降)| ✅ **本文证实**(内部 1.40x@0.50) |
| ③ 诚实标注地板 | ✅ 已写入 docs + 报告器 theory-vs-wall |

base OpenVLA 上的 claim 已收敛完毕:**选择质量 + 机制有效已证,端到端 wall-clock 加速
受 decode 地板限制,显式 defer 到 prefill 主导的 OFT/pi0。**

## 5. 后续

- **次高杠杆 = 解锁 OFT 权重**:端到端加速 claim 唯一的真实阻塞(盘上只有 base)。
  拿到 OFT 后,这条同样的 prefill 曲线会直接转成 wall 曲线。
- 可选:若要在 base 上压低后端固定开销,需在核心代码层动 internal forward
  (index_select/cache 重建),那超出"只读验证 + 报告对齐"的范围,需另行确认。
- 最优工作点:base + internal-at-layer-2 下 **ρ=0.50** 是当前 prefill 最优点。
