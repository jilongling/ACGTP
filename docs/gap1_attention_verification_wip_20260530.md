# 缺口①验证(真实注意力 vs QK代理)— 进行中交接 (2026-05-30)

> 这是一份 **未完成任务的交接文档**。新对话从这里接着做。
> 任务:验证 ACGTP 设计宪法第2层"True LLM Attention Verification"当前是半成品
> (用 QK 代理而非真实物化注意力),并定量评估这个缺口是否影响成功率。

## 0. 背景:为什么做这件事

方案体检发现三个缺口(见 acgtp-progress 记忆 + status doc)。**缺口① = 最大缺口:**
设计宪法第2层要求"读真实 LLM 内部注意力(P_sem/P_act)",但代码实测
`internal_attention_source = llm_qk_text_to_vision` —— 走的是 QK 投影代理
(`pruning/internal_pruning.py:305 _qk_text_to_visual_attention`),不是物化注意力
(`pruning/internal_pruning.py:262 _text_to_visual_attention`)。

根因(已确认,代码级):`quota_config` 里没有 `try_output_attentions` 这个 key
(`pruning/hook.py:820-846`),所以 `internal_pruning.py:993` 的 `try_materialized_attn`
恒为 False，真实注意力分支永不执行。而且实测 `geo_attention_iou=0.084`
(几何与代理高度不一致)。

## 1. 已确认的硬事实(不依赖探针跑完)

**模型跑 FlashAttention,真实注意力矩阵默认不可得。**
- eval loader `use_flash_attention=True`;FlashAttention 不物化 attention 概率。
- 探针第一次跑出 `real_attention_available=false` —— 即使克隆该层为 eager attention
  强制 `output_attentions=True`，`_text_to_visual_attention` 仍判 false（eager 重放
  路径的 attention_mask 形状 / RoPE position 没对上）。
- 结论:当年留 QK 代理做 fallback **不是偷懒,是 FlashAttention 的硬约束**。
  要拿真实注意力有两条路,都不便宜:
  (a) 整模型切 eager/sdpa → 能拿到但推理变慢、显存涨,违背"加速"初衷;
  (b) 只在 prune_layer 那层切 eager → 可行但需正确重建 attention_mask + RoPE,
      属于核心 forward 改动(需用户确认才能动)。

## 2. 已做的东西

**新探针(纯只读,不改任何核心剪枝代码):**
`scripts/probe_attention_proxy_gap.py`
- 加载 base OpenVLA,取一帧 LIBERO 观测,跑一次 predict_action。
- 在 prune_layer(默认2)处算三个 text→vision 重要性向量:
  (a) QK代理(backend 实际用的函数)
  (b) 真实物化注意力(临时克隆该层为 eager LlamaAttention,进程内,不改文件)
  (c) 几何 geo_soft_score(从捕获的 plan.geometry_payload 取)
- 输出:三者两两 top-k IoU、spearman 秩相关、各自对 geo_protect token 的 top-k 覆盖。
- 用 in-process monkeypatch 包裹 `backend.resolve_visual_keep_indices` 捕获 plan
  几何 payload；用 forward_pre_hook 捕获 prune_layer 输入 hidden_states。两者都在
  finally 里还原/移除,scoped 到本进程。

**已修的 bug:** 第一次跑捕获到的是 decode 步(seq_len=1)而非 prefill 步,导致三向量
全空。已改 `_capture_layer_input` 只在 seq_len>1 且首次时捕获(保留 prefill)。
**改完已 py_compile OK,但因平台分类器故障还没成功重跑。**

**第一次跑的(有 bug 的)产物:** `outputs/attention_proxy_gap/attention_proxy_gap.json`
—— seq_len_at_prune_layer=1（decode 步,无效）, real_attention_available=false,
geo_protect_count=52, geometry 自覆盖=1.0。这份要被下次重跑覆盖。

## 3. 下一步(新对话接着做)

**立即可跑(纯只读,1-2 分钟,会占 GPU):**
```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python scripts/probe_attention_proxy_gap.py \
  --prune_layer 2 --keep_ratio 0.50 --output_dir outputs/attention_proxy_gap
```
读 `outputs/attention_proxy_gap/attention_proxy_gap.json`,确认:
- `seq_len_at_prune_layer` 应为 291(prefill),不再是 1。
- `topk_iou` / `spearman` 应非空。
- 重点看 `topk_iou["qk_proxy__vs__geometry"]` 和 `geo_protect_topk_coverage`。

**要回答的问题:**
1. QK代理相比纯几何到底多选对了什么(IoU 越低 = 代理带来越多"几何看不到"的 token)。
2. 真实注意力如果还是 false,说明 base + FlashAttention 下第2层注意力**事实上拿不到**,
   那 base 阶段就应诚实声明"用 QK 代理",真实注意力推广到 OFT/eager 模型时再上。
3. 代理/真实注意力对 geo_protect 的覆盖低不低 —— 低则说明注意力信号会和几何硬保护
   打架,补它的优先级要重新评估。

**注意环境:** 必须用 `/infini-data/miniconda3/envs/openvla/bin/python`
(torch 2.2.0+cu121, libero OK)。默认 python 没有 torch。

## 4. 完成度评估(体检结论,供参考)

| 设计宪法三层 | 状态 |
|---|---|
| 第1层 几何先验(P_geo+soft)| ✅ 完整(geo_protect_mask 实装,critical=0)|
| 第2层 真实 LLM 注意力验证 | ⚠️ **半成品**（QK 代理,非物化）= 缺口① |
| 第3层 风险自适应内部剪枝 | ✅ 基本完整（coverage-based risk）|
| 即插即用（跨模型）| ⚠️ OpenVLA 硬编码 image_start=1/len=256（hook.py:1448-1449）= 缺口② |

base 上"选择质量+机制"验证 ≈ 90% 可用；完整 ACGTP 方案 ≈ 60-65%。
缺口①（真实注意力）、缺口②（跨模型抽象层）是推广到其他 VLA 前必补的两块。
缺口③（端到端加速）受 base decode 地板限制,解法是解锁 OFT 权重（外部依赖）。

## 5. 相关文档/数据
- 根因 + 现状:`docs/acgtp_status_20260530.md`
- 5篇论文综述:`docs/vla_efficient_inference_survey_20260530.md`
- 指标对齐:`docs/metric_alignment_report_20260530.md`
- prefill曲线（断言②已证）:`docs/prefill_retention_sweep_20260530.md`
- 设计宪法 + 进度:项目记忆 `acgtp-final-design.md` / `acgtp-progress.md`
