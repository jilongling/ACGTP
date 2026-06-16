# VLA 高效推理论文综述 — 加速机制拆解 (2026-05-30)

本文整理五篇 VLA 视觉 token 高效推理论文，**核心问题不是"怎么选 token"，
而是"选完之后靠什么把 token 减少转成真实 wall-clock 节省"**。这直接决定了
ACGTP 在 base OpenVLA 上为什么不加速（见 `acgtp_status_20260530.md` 的根因），
以及把机制搬到哪里才能发挥（路线 A / B）。

所有数字与论断均来自各论文 arXiv 全文核对，非凭记忆。

## 0. 一句话结论

加速能否兑现，取决于两件事的组合，与"选得准不准"几乎无关：

1. **砍在网络的哪个位置** —— 第 0 层之前（所有层受益）vs 中间层 K（只有 K 之后受益）。
2. **在哪种解码模式上测** —— prefill 主导 / 并行解码（OFT、π0、diffusion）能兑现；
   自回归 decode 主导（base OpenVLA）几乎兑现不了。

base OpenVLA 是"自回归 + decode 主导"，是这五篇里**最难加速的场景**；论文要么
绕开它（测 OFT/π0），要么换机制（跨帧 KV 复用）。

## 1. 三条"省时间"的机制路径

把五篇按节省机制归类，而非按选择标准：

### 路径 A：在 LLM 第 0 层之前剪 / 合并（缩短进入 LLM 的序列）
缩短后的序列让**全部 transformer 层**都跑短，prefill 和后续 decode 的 KV 都变小。
这是最干净的省法。代表：VLA-ADP、VLA-IAP、DepthCache。

### 路径 B：在 LLM 中间层 K 剪（FastV 式）
只有第 K..L 层跑短；**第 0..K-1 层永远跑满**。节省按层数比例打折，且要付
index_select / causal_mask 重建 / KV cache 跨层不一致的开销。代表：VLA-Pruner（K=3）。
**ACGTP internal 模式就在这条路径上（K=2），且测在最差的 base 上。**

### 路径 C：跨时间步复用 KV（不缩短序列）
不动序列长度，把"帧间几乎不变"的静态 token 的 K/V 直接继承上一帧，跳过它们在
decoder 层的 QKV 投影 + MLP。代表：VLA-Cache。这是唯一正面应对自回归 decode 的，
但它自己承认只省到每步的第一次（prefill-like）前向。

## 2. 逐篇拆解

### 2.1 VLA-ADP — Action-aware Dynamic Pruning (arXiv 2509.22093)
- **选择**：embedding 阶段用 LLM 第 0 层的 Q/K 投影算 text→vision 相关性，按保留率 ρ
  取 top-k（主/腕相机按权重 α 分配，如 4:6）。再叠一个**按末端执行器运动幅度的动态门控**：
  精细阶段（抓取/放置，运动小）关闭剪枝保全视野，粗略移动阶段（运动大）才剪。
- **省时间机制（路径 A）**：论文明说"pruning happens before the LLM at the embedding stage…
  all H layers operate on S′"。砍在第 0 层之前，**全部 32 层跑短**，不是逐层渐进丢。
  自适应是**逐时间步（逐 action-window）**，不是逐层。
- **测在哪**：**OpenVLA-OFT（并行解码，chunk=8）**。OFT 没有自回归 decode 循环，
  整个动作生成就是一次前向，视觉 token 正好是其主成本。
- **数字**：LIBERO 上 30% 保留 = 1.35x（LLM 侧），50–70% 保留成功率掉 ≤0.9%；
  真机 Jaco2 76.9→51.8ms = 1.49x。
- **对 ACGTP 的启示**：这是路线 A（砍在 projector 前）+ 路线 B（测 OFT）的"标准答案"，
  且它的动态门控逻辑和 ACGTP 的 risk-adaptive 思路高度相似。

### 2.2 VLA-IAP — Interaction-Aligned Pruning (arXiv 2603.22991)
- **注意**：IAP = **Interaction-Aligned**（交互对齐），不是 importance-aware。
- **选择**：从"感知优先"（语义注意力）转向"交互优先"。加一个 **Sobel 式几何/边缘先验**
  （捕捉语义注意力会漏掉的细把手、透明边缘），再加**语义-运动对齐调度**：算语义 mask
  与运动 mask 的 IoU，任务早期两者不一致时只保守剪背景，"交互锁定"（高 IoU）后才激进剪。
- **省时间机制（路径 A）**：选中的 token gather 成紧凑序列后"concatenated with the text…
  fed into the subsequent LLM"；DreamVLA 上则是在 vision encoder 输出、GPT-2 backbone 之前
  应用 mask。**砍在 LLM 第 0 层之前**，每层都受益。
- **测在哪**：**OFT / π0 / π0.5 / DreamVLA**，全是 prefill 主导 / 并行解码 / flow-matching。
  自回归 base OpenVLA 只作为 Table 2 的参考行，**不施加剪枝**。
- **数字（A100）**：OFT/LIBERO 70% 保留 1.25x（97.8% SR）、50% 1.37x、30% 1.54x。
  真机 π0.5 单/双臂 1.48x/1.47x。
- **对 ACGTP 的启示**：和 ACGTP 的"几何先验 + 不能只靠语义注意力 + IoU 仅作诊断"几乎是
  同一套设计哲学，但它把机制落在 prefill 主导模型上才拿到加速。**最值得对标的工作。**

### 2.3 VLA-Pruner (arXiv 2511.16449)
- **选择**：双目标重要性 = prefill 语义注意力 + action-decode 注意力（当前步用 EMA 时间平滑
  估计，因为 prefill 时拿不到真 decode 注意力），再 "Combine-then-Filter"（两个 top-k 取并集
  后做多样性/mRMR 过滤）。
- **省时间机制（路径 B，FastV 式中间层剪）**：在 **LLM 第 K=3 层**丢 token。论文复杂度分析
  明说"the first K−1 layers still use the full sequence μ, while the remaining T−K+1 layers use
  the shortened sequence"。**第 0–2 层永远跑满**；只剪一次；省的是第 3–31 层的 prefill +
  这些层带进 decode 的更短 KV。
- **测在哪**：三种都测——自回归 OpenVLA、OFT、π0（flow-matching）。
- **数字（RTX 4090）**：头条 **1.99x 在 OFT@12.5% 保留**（135.78→68.95ms）；
  **base OpenVLA@12.5% 只有 1.83x**（236.41→129.01ms）；50% 保留时 OpenVLA 1.33x / OFT 1.46x。
- **关键反证（直接印证 ACGTP 根因）**：base 上 FLOP 砍到 ~30%（理论 ~3.3x），wall 只 1.83x。
  这 ~2x 的"FLOP 省了但时间没省"缺口，正是 decode 地板：6 次串行 decode 被权重带宽 bound +
  前 K 层跑满，token 剪枝碰不到。**ACGTP 与它同在路径 B，但 ACGTP 还只测 base、K=2，所以更差。**

### 2.4 VLA-Cache — Adaptive Token Caching (arXiv 2502.02175)
- **选择**：利用闭环操作中相邻帧的时间连续性，找帧间几乎不变的视觉 token；cross-attention 过滤器
  强制"任务相关"的静态 token（夹爪、目标）仍重算；逐层按注意力熵调节复用比例。
- **省时间机制（路径 C，跨帧 KV 复用）**：时刻 t 对静态 + 任务无关 token，**继承 t−1 缓存的 K/V**，
  不再过 decoder 层 —— 跳过的是该 token 的 **QKV 投影 + MLP**。靠 Transformer 置换不变性，
  部分更新仍得到合法注意力。节省发生在**语言 decoder 内部**，不是 vision encoder / projector。
- **直面自回归 decode 的诚实声明**：Appendix D 明说"largest gain occurs when generating the
  **first** action token… subsequent tokens are decoded autoregressively **without additional cost**"。
  即：**6 次串行 decode 一步都没省**，增益全在每步第一次（prefill-like）前向。它还指出 FastV/SparseVLM
  在 VLA 上"fail to improve inference speed"，因为动作输出只有 7 个 token。
- **测在哪 / 数字（RTX 4090）**：OpenVLA/LIBERO 51.91→31.83ms（**1.63x**，SR 75.0→74.7）；
  OFT 79.05→62.59ms（~1.26x，控制频率 65→79Hz）；CogACT ~1.37x。
- **对 ACGTP 的启示**：这是 base OpenVLA 上唯一拿到显著加速（1.63x）的路线，因为它**换了机制**
  （缩序列→复用 KV），把节省集中到那一次重前向。若坚持留在 base，这是机制层面的另一种出路。

### 2.5 DepthCache — Depth-Guided Token Merging (arXiv 2603.10469)
- **选择**：训练-free 的 token **合并**（非剪枝、非 KV 复用）。用深度图分区，远/背景区高合并率、
  近场工作区低合并率；双重保护集（LLM cross-attention + 深度梯度边缘）护住关键 token；
  合并过程"跨连续帧分摊"以保时序平滑。
- **省时间机制（路径 A，缩序列）**：名字虽叫 Cache，但**无跨帧 KV 复用**（"Cache"是类比 DeepCache
  的品牌名）。靠 bipartite soft matching 把相似 patch embedding 合并（按 size 加权平均），
  **更少的 token 进 LLM backbone**。双相机 512→稳态 ~300。完全在 vision encoder 之外操作。
- **测在哪 / 数字（RTX 4090）**：π0.5 1.28x（−0.3% SR）、OpenVLA 1.21x（−1.0%）、
  GR00T-N1 2.2B 1.07x。对照：FastV −12.7% SR、ToSA 0.94x（不加速还掉 24% SR）。
  真机 PIPER 1.33x。
- **关键观察**：自回归 base OpenVLA 只 1.21x，最小的 GR00T-2.2B 只 1.07x —— **越是 decode-bound /
  backbone 越小，token 砍了越不值钱**。再次印证：加速主要来自 prefill 主导模型。

## 3. 横向对照表

| 论文 | 砍在哪 | 机制路径 | 实际省的计算 | 测的模型/解码 | 加速 | 自回归 base 表现 |
|---|---|---|---|---|---|---|
| VLA-ADP | 第0层前 | A 缩序列 | 全 32 层跑短 | OFT 并行 | 1.35x / 真机1.49x | 不在 base 上施加 |
| VLA-IAP | 第0层前 | A 缩序列 | 全层跑短 | OFT/π0/π0.5/Dream | 1.25–1.54x | 仅参考行，不剪 |
| VLA-Pruner | 中间层 K=3 | B 半受益 | 第3–31层+更短KV | OFT(头条)/base/π0 | OFT 1.99x | base 1.83x（FLOP 3.3x→wall 1.83x） |
| VLA-Cache | decoder 内 | C KV复用 | 静态token的QKV+MLP | OpenVLA/CogACT/OFT | base 1.63x | **1.63x（唯一 base 显著加速）** |
| DepthCache | 第0层前 | A 合并 | 更短序列进backbone | π0.5/OpenVLA/GR00T | 1.21–1.28x | base 1.21x |

## 4. 对 ACGTP 的直接结论

1. **三条铁律**：(a) 要砍在第 0 层之前（路径 A）才能让所有层受益；中间层剪（路径 B）只半受益且有
   额外开销。(b) 真实大加速几乎全来自 prefill 主导 / 并行解码模型。(c) 没有任何论文能用"砍视觉 token"
   加速自回归 decode —— VLA-Cache 唯一直面它，靠的是换成跨帧 KV 复用。

2. **ACGTP 现状 = 最差组合**：internal-at-layer-2（路径 B）+ 自回归 base（最难场景）+ K=2，
   三条逃生路线一条没占。-1.95ms 是机制天花板的忠实读数，不是指标 bug（CUDATimer 完整覆盖
   prefill+6×decode，LM hook 跨调用累加，probe 已能拆分 prefill/decode；详见 `acgtp_status_20260530.md` §4）。

3. **逃生路线对应关系**：
   - **路线 A（砍到 projector 前）** = 走 ADP/IAP/DepthCache 的路径 A，去拿 base 上那 ~20% prefill。
   - **路线 B（换 OFT/pi0）** = 走 ADP/IAP/Pruner 共同的"测 prefill 主导模型"，让那 20% 变成主成本。
   - **第三条（若坚持留在 base 且要显著加速）** = 走 VLA-Cache 的路径 C：跨帧 KV 复用，而非缩序列。
     当前 ACGTP 设计未覆盖，仅作备选记录。

4. **最值得对标**：VLA-IAP（几何先验 + 交互对齐 + IoU 仅诊断，与 ACGTP 宪法几乎同源）与
   VLA-ADP（运动门控的动态保留，与 ACGTP risk-adaptive 同源）。两者都把同样的设计哲学落在了
   prefill 主导模型上才兑现加速 —— 这正是 ACGTP 当前缺的最后一块拼图。

## 5. 来源

- VLA-ADP: https://arxiv.org/abs/2509.22093 （Action-aware Dynamic Pruning）
- VLA-IAP: https://arxiv.org/abs/2603.22991 （Interaction-Aligned Pruning）
- VLA-Pruner: https://arxiv.org/abs/2511.16449 ; 代码 https://github.com/MINT-SJTU/VLA-Pruner
- VLA-Cache: https://arxiv.org/abs/2502.02175
- DepthCache: https://arxiv.org/abs/2603.10469
- 参照：OpenVLA-OFT https://arxiv.org/abs/2502.19645 （解释 OFT 为何 prefill 主导）

相关项目文档：`acgtp_status_20260530.md`（根因 + 路线 A/B 决策）、`eval_protocol.md`（评估协议）。
设计宪法与进度在项目记忆：acgtp-final-design / acgtp-progress。
