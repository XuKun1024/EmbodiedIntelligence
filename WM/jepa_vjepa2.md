# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

> **作者**：Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Franziska Meier, Yann LeCun, Michael Rabbat（共同末位）, Nicolas Ballas（共同末位）  
> **机构**：FAIR at Meta；Mila – Quebec AI Institute and Polytechnique Montréal  
> **年份/会议**：2025 / 技术报告（arXiv）  
> **arXiv**：arXiv:2506.09985

## 一句话总结

V-JEPA 2 将 V-JEPA 扩展至 10 亿参数和超过 100 万小时视频，并在冻结的视频编码器基础上，仅用 62 小时机器人视频训练出 action-conditioned 世界模型（V-JEPA 2-AC），实现了在两个不同实验室的 Franka 机械臂上的零样本操控规划，证明表示空间中的预测与规划既高效又有效。

## 核心贡献

- **大规模自监督视频预训练的扩展**：将 V-JEPA 框架扩展至 10 亿参数（ViT-g）和超过 100 万小时视频（VideoMix22M），引入渐进式分辨率训练策略，在运动理解（SSv2 77.3 top-1）和视频问答上达到 SOTA。
- **Action-Conditioned World Model（V-JEPA 2-AC）**：在冻结的 V-JEPA 2 encoder 基础上，仅用 62 小时无标签机器人视频（Droid 数据集），训练出 300M 参数的动作条件化世界模型，无需任务特定训练即可用于机器人规划。
- **表示空间中的高效预测与规划**：证明在 representation space（而非 pixel space）中进行预测和规划既高效又有效——V-JEPA 2-AC 每步规划仅需 16 秒，而对比的 Cosmos 视频生成模型需要 4 分钟。
- **零样本机器人操控**：在两个不同实验室的 Franka 机械臂上零样本部署，实现抓取、持物移动、pick-and-place 等灵巧操控任务，这些实验室均未出现在训练数据中。
- **视频编码器与 LLM 对齐达到 VideoQA SOTA**：首次证明一个无语言监督预训练的视频编码器，通过与 LLM（Llama 3.1 8B）对齐后，在 8B 参数量级的 VideoQA 任务上可达到 SOTA，挑战了"需要语言监督才能做好 VidQA"的传统认知。

## 方法

### Enc()

**架构**：Vision Transformer (ViT)，三个尺度：

| 模型 | 参数量 | Width | Depth | Heads | MLP |
|------|--------|-------|-------|-------|-----|
| ViT-L | 300M | 1024 | 24 | 16 | 4096 |
| ViT-H | 600M | 1280 | 32 | 16 | 5120 |
| ViT-g | 1B | 1408 | 40 | 22 | 6144 |

**Patch 大小**：16×16，使用 2×16×16 的 tubelet（时间维度步长为 2）。

**视频输入处理方式**：
- 视频被 patchify 成一系列 tubelets，大小为 2×16×16（T×H×W）。
- 位置编码：使用 3D 扩展的 1D-RoPE（**3D-RoPE**），将特征维度分为三段（时间、高度、宽度），分别应用 1D 旋转位置编码。相比绝对 sincos 位置编码，3D-RoPE 有助于稳定大模型训练。
- 主训练阶段：16 帧，256×256 分辨率，4fps。
- Cooldown 阶段：64 帧，[256, 384, 512] 分辨率（渐进式分辨率训练）。
- 图像作为 16 帧相同内容的视频处理（时间复制）。

**预训练数据集 VideoMix22M（VM22M）**：

| 数据集 | 样本量 | 时长 | 采样权重 |
|--------|--------|------|---------|
| SSv2 | 168K | 168 小时 | 0.056 |
| Kinetics (K400/600/700) | 733K | 614 小时 | 0.188 |
| HowTo100M | 1.1M | 134K 小时 | 0.318 |
| YT-Temporal-1B（检索式 curation） | 19M | 1.6M 小时 | 0.188 |
| ImageNet | 1M 图像 | — | 0.250 |
| **合计** | **22M** | **超过 100 万小时** | — |

### Pred()

**预训练阶段 Predictor（action-free）**：

- 架构：ViT-small（22M 参数），12 层，12 头，384 维，MLP 1536。
- 预测目标：在 embedding space 中预测被 mask 掉的视频 patch 的表示，而非像素。目标由 EMA encoder（指数滑动平均的 encoder 权重）计算。
- 训练目标（L1 loss）：
$$\min_{\theta,\phi,\Delta y} \|P_\phi(\Delta y, E_\theta(x)) - \text{sg}(E_{\bar\theta}(y))\|_1$$
- Mask 策略：多块掩码（multiblock masking），与 V-JEPA 原始工作相同。空间 mask scale [0.15, 0.7]，时间 mask scale [1.0, 1.0]，mask 宽高比 [0.75, 1.5]。
- **是否 action-conditioned**：**否**。预训练阶段的 Predictor 不使用动作条件，是纯自监督的特征预测。

**V-JEPA 2-AC：Action-Conditioned World Model（post-training 阶段）**：

这是 V-JEPA 2 的关键扩展，在预训练完成后进行 post-training，encoder 权重冻结不更新。

- 架构：300M 参数 Transformer 网络，24 层，16 头，1024 维隐藏层，GELU 激活函数。
- 预测目标：自回归地预测未来视频帧的表示，条件化于过去视频帧的表示（由冻结的 V-JEPA 2 encoder 编码）、机器人控制动作（action）和末端执行器状态（end-effector state）。

**Action 表示**：
- 每个 action $a_k$ 是 7 维实值向量，表示相邻帧间末端执行器状态的变化（delta）。
  - 前 3 维：笛卡尔位置变化（Δx, Δy, Δz）
  - 中 3 维：姿态变化（extrinsic Euler angles）
  - 末 1 维：夹爪状态变化

**End-effector state 表示**：
- 每个 $s_k$ 是 7 维实值向量，相对于机器人基座定义（位置 3 维 + 姿态 3 维 + 夹爪状态 1 维）。

**Action Conditioning 机制**：
- Encoder 输出的每帧特征图 $z_k \in \mathbb{R}^{H \times W \times D}$（ViT-g 编码后形状为 16×16×1408），每帧独立编码。
- Action、end-effector state 和 flattened 特征图分别经过独立的可学习仿射变换（affine transformation）映射到 predictor 的隐藏维度。
- Predictor 使用 **block-causal attention pattern**：每个时间步的 patch 特征可以 attend 到同一时间步及之前时间步的 action、end-effector state 和其他 patch 特征。
- 对视频 patch 使用 3D-RoPE 表示时空位置，对 action 和 pose token 仅使用时间维度的旋转位置编码。

**双重训练损失**：
1. **Teacher Forcing Loss**：Predictor 以当前帧真实表示为输入，预测下一时间步表示。
2. **Rollout Loss**：将 Predictor 的输出反馈作为下一步输入，训练多步预测能力，减少 autoregressive rollout 中的误差累积。

两种损失的和共同优化，均使用 L1 损失。

**训练数据**：Droid 数据集中少于 **62 小时**的无标签机器人视频（4 秒片段，256×256，4fps，16 帧），仅使用左外视角摄像头。

**规划方式（推断阶段）**：使用 Cross-Entropy Method（CEM）在表示空间中优化动作序列，目标是最小化世界模型预测的未来状态与目标图像表示之间的 L1 距离。

### Dec()

**V-JEPA 2 预训练阶段**：无 Decoder。这是 JEPA 的核心设计——在表示空间预测，不重建像素。

**V-JEPA 2-AC 推断可视化**：为可视化目的，在 Droid 数据集上额外训练了一个轻量级帧解码器（feedforward 网络，MSE 像素重建损失），但这不是模型的一部分，仅用于解释性分析，不参与世界模型训练。

### 关键设计亮点

1. **渐进式分辨率训练（Progressive Resolution Training）**：主训练阶段使用 16 帧/256×256，cooldown 阶段切换到 64 帧/384×384，实现 8.4 倍加速，同时保留高分辨率长视频的性能收益。这使得在超大规模数据上训练 1B 参数模型成为可行。

2. **3D-RoPE 位置编码**：将 1D-RoPE 扩展至时间、高度、宽度三个维度，稳定大模型训练，同时支持推断时处理不同分辨率和帧率的视频。

3. **表示空间规划的高效性**：V-JEPA 2-AC 在 latent space 中使用 CEM 规划，每步仅需 16 秒，而基于像素生成的 Cosmos（7B 参数）需要 4 分钟，速度提升约 15 倍，且性能全面优于 Cosmos。

## 实验结果

### 视频理解（Frozen Encoder + Attentive Probe）

| 模型 | 参数量 | 平均（6任务） | SSv2 | K400 | IN1K |
|------|--------|-------------|------|------|------|
| V-JEPA ViT-H（原版） | 600M | 85.2 | 74.3 | 84.5 | 80.0 |
| V-JEPA 2 ViT-L | 300M | 86.0 | 73.7 | 85.1 | 83.5 |
| V-JEPA 2 ViT-H | 600M | 86.4 | 74.0 | 85.3 | 83.8 |
| V-JEPA 2 ViT-g | 1B | 87.5 | 75.3 | 86.6 | 84.6 |
| **V-JEPA 2 ViT-g₃₈₄** | **1B** | **88.2** | **77.3** | **87.3** | **85.1** |
| DINOv2 ViT-g | — | 81.1 | — | — | — |
| SigLIP2 | — | 81.1 | — | — | — |

### 动作预测（EK100 Action Anticipation，recall@5）

| 方法 | Verb | Noun | Action |
|------|------|------|--------|
| PlausiVL（前 SOTA，8B） | 55.6 | 54.2 | 27.6 |
| **V-JEPA 2 ViT-g₃₈₄** | **63.6** | **57.1** | **39.7** |

V-JEPA 2 ViT-g₃₈₄ 相比 PlausiVL 提升 +12.1 点（44% 相对提升）。

### Video Question Answering（8B 参数量级，V-JEPA 2 ViT-g₃₈₄ + Llama 3.1 8B）

| 基准 | V-JEPA 2 | PLM 8B（前 SOTA） | 提升 |
|------|----------|-----------------|------|
| PerceptionTest（test acc） | **84.0** | 82.7 | +1.3 |
| MVP（paired acc） | **44.5** | 39.7 | +4.8 |
| TempCompass（multi-choice） | **76.9** | 72.7 | +4.2 |
| TemporalBench（short-QA） | **36.7** | 28.3 | +8.4 |
| TOMATO（acc） | **40.3** | 33.2 | +7.1 |
| TVBench（acc） | 60.6 | **63.5** | -2.9 |
| MVBench（acc） | 73.5 | **77.1** | -3.6 |

### 零样本机器人操控（两个实验室 10 次试验平均成功率）

| 方法 | Reach | Grasp Cup | Grasp Box | Reach w/ Cup | Reach w/ Box | P&P Cup | P&P Box |
|------|-------|-----------|-----------|--------------|--------------|---------|---------|
| Octo（VLA，BC） | 100% | 15% | 0% | 15% | 70% | 15% | 10% |
| **V-JEPA 2-AC** | **100%** | **65%** | **25%** | **75%** | **75%** | **80%** | **65%** |

### 规划效率对比（Lab 2）

| 方法 | 样本数 | 每步规划时间 | Grasp Cup | P&P Cup |
|------|--------|------------|-----------|---------|
| Cosmos（视频生成，7B） | 80 | **4 分钟** | 0% | 0% |
| **V-JEPA 2-AC** | 800 | **16 秒** | 60% | 80% |

V-JEPA 2-AC 使用 10 倍更多样本，每步规划时间仅为 Cosmos 的 1/15，且性能全面优于 Cosmos。

### 消融实验关键数字

| 改进 | 平均提升 |
|------|---------|
| 数据扩展（VM2M → VM22M） | +1.0 |
| 数据 curation（uncurated → curated） | +1.4 |
| 模型扩展（ViT-L → ViT-g） | +1.5~1.7 |
| 延长训练（90K → 252K 步） | +0.8 |
| 渐进式分辨率（cooldown 加长视频） | +0.7 |
| 评估时延长视频（16 → 64 帧） | +9.7 |

## 局限性

- **对摄像头位置敏感**：V-JEPA 2-AC 在没有显式相机标定的情况下，必须从单目 RGB 图像中隐式推断动作坐标轴。当机器人底座不在画面中时，推断坐标轴不唯一，导致世界模型误差。论文通过手动调整摄像头位置来规避此问题，推断坐标轴的旋转误差几乎是摄像头位置的线性函数。
- **长时域规划受限**：Autoregressive 预测存在误差累积，随着 rollout 步数增加，表示预测精度下降；搜索空间随规划步长指数增长；当前方法依赖子目标图像（sub-goal images）才能完成 pick-and-place，不能从单一最终目标图像完成。
- **需要图像目标，不支持语言指令**：当前系统需要视觉目标图像，无法接受语言指令，而在实际部署中语言目标更自然。
- **预测时域受限**：当前聚焦于约 16 秒以内的预测，对于更复杂的长时域任务需要进一步建模创新。
- **模型规模仍有扩展空间**：当前最大仅扩展至 1B 参数，前人研究已探索至 20B 参数的视觉编码器，V-JEPA 2 在更大规模上的扩展行为仍未知。
- **动作预测能力主要来自语义表示而非时序预测**：消融显示仅用 encoder 达到 39.1 recall@5，encoder+predictor 达到 39.7，仅用 predictor 仅达到 20.2，说明预测能力主要来自表示的语义丰富性。

## 与 JEPA 系列演进关系

V-JEPA 2 是 JEPA 系列的**第三步**，同时沿两个维度大幅扩展：

- **继承自 V-JEPA**：embedding space 中的特征预测、EMA 目标编码器、stop-gradient、多块时空掩码策略、无 Decoder 的设计哲学、Attentive Probing 评估协议。
- **对 V-JEPA 的扩展（规模）**：参数量从 630M 扩展至 1B，预训练数据从 200 万视频扩展至 2200 万视频（100 万小时），引入 3D-RoPE 和渐进式分辨率训练以支持大规模训练。
- **对 V-JEPA 的扩展（功能）**：引入 V-JEPA 2-AC，将纯自监督视觉表示学习扩展至 action-conditioned 世界模型，实现机器人规划。这是 JEPA 系列首次明确用于机器人具身智能场景。
- **与 I-JEPA 的关系**：V-JEPA 2 预训练阶段继承了 I-JEPA 的所有核心原则（embedding space 预测、无 Decoder、EMA 防坍塌），并将其扩展至视频和超大规模。
- **与 LeWM 的关系**：LeWM 与 V-JEPA 2-AC 共享 action-conditioned latent world model 的核心思路，但 LeWM 采用**端到端训练**（encoder 不冻结），并以 SIGReg（可证明的反坍塌保证）替代 EMA/stop-gradient，在极小规模（15M 参数）下实现竞争性规划性能。
