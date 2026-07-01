# Revisiting Feature Prediction for Learning Visual Representations from Video

> **作者**：Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran（共同末位）, Nicolas Ballas（共同末位）  
> **机构**：FAIR at Meta；Inria；École normale supérieure / CNRS / PSL；Univ. Gustave Eiffel；Courant Institute / CDS at NYU  
> **年份/会议**：2024 / 技术报告（arXiv）  
> **arXiv**：arXiv:2404.08471

## 一句话总结

V-JEPA 将 I-JEPA 的表示空间预测思想扩展到视频域，通过预测被时空多块掩码遮挡区域的特征表示（而非像素），以更少的训练数据和迭代次数，在运动理解（SSv2）和外观理解（K400）任务上同时超越所有像素重建基线，证明特征预测可作为从视频中无监督学习视觉表示的独立有效目标。

## 核心贡献

- **特征预测作为独立目标的有效性验证**：证明 feature prediction（而非 pixel reconstruction）可以作为从视频中无监督学习视觉表示的独立有效目标，无需预训练图像编码器、文本、负样本、重建或其他监督信号。
- **多功能视觉表示**：V-JEPA 学到的表示在 frozen backbone（不微调参数）条件下同时适用于运动理解任务（SSv2）和外观理解任务（K400），在 SSv2 上比其他方法高出 +6% 以上。
- **特征预测优于像素预测**：在 frozen evaluation 协议下，特征预测始终优于像素预测；在 fine-tuning 下具有竞争力，且训练时间显著更短（约 2 倍加速，处理样本量少一个数量级，仅 270M vs. 2400M）。
- **更高的标签效率**：随着标注样本减少，V-JEPA 与 pixel prediction 方法的性能差距扩大。在 K400 上，V-JEPA 在 5% 标签下仍有 68.2%，而 VideoMAEv2 仅 37.0%。
- **3D 多块掩码策略（Multi-Block Masking）**：提出针对视频的时空掩码策略，通过多个连续空间块在整个时间维度延伸来防止信息泄漏，平均掩码率约 90%，并通过消融实验系统验证了各设计选择。

## 方法

### Enc()

**架构**：Vision Transformer (ViT)，用作 x-encoder（上下文编码器）和 y-encoder（目标编码器，为 x-encoder 的 EMA），两者结构相同。

**训练的三个模型变体**：

| 模型 | 参数量 | 输入分辨率 | Patch 大小 |
|------|--------|-----------|-----------|
| ViT-L/16 | ~200M | 224×224 | 16×16 |
| ViT-H/16 | ~630M | 224×224 | 16×16 |
| ViT-H/16₃₈₄ | ~630M | 384×384 | 16×16 |

**视频输入处理方式（时空处理）**：

- 输入：随机采样 16 帧，时间步长（temporal stride）为 4，覆盖总共 64 帧（约 2-3 秒，30fps 视频约 2 秒）。
- **Tokenization**：使用一个 3D 卷积，包含 $d$ 个大小为 `2×16×16` 的滤波器，时间步长为 2，空间步长为 16。
  - 输出 tensor shape：`8 × 14 × 14 × d`
  - 加入 3D sin-cos 绝对位置编码后展平为 1D 序列：`1568 × d`（即 L=1568 个 spatio-temporal tokens）
- **Tubelet size**：每个 token 对应 2 帧 × 16×16 像素的时空块（tubelet_size=2）。
- 预训练时不使用 [CLS] token；掩码操作通过直接丢弃 token 实现（drop tokens），而非用 mask token 替换。

**EMA 目标编码器**：y-encoder 是 x-encoder 的指数移动平均，初始 momentum=0.998，线性增加至 1.0，结合 stop-gradient 防止表示坍塌。

### Pred()

**预测目标**：在 embedding space（特征空间）中预测被遮挡时空区域的表示向量。Predictor 输出每个 masked patch 对应的 $d$ 维向量，回归到 y-encoder 的输出（L1 loss）。

**Predictor 架构**：窄型（narrow）Vision Transformer。
- 12 个 transformer blocks
- embedding dimension：384
- self-attention heads 数量与对应 backbone 相同

**Predictor 的输入**：
1. x-encoder 输出的可见区域 token 序列（已编码的上下文表示）
2. 一组可学习的 mask tokens：参数化为共享可学习向量 + 3D sin-cos 绝对位置编码（指示被遮挡位置）

**损失函数**：L1 regression（比 L2/MSE 更稳定）：
$$\min_{\theta,\phi} \|P_\phi(E_\theta(x), \Delta y) - \text{sg}(\bar{E}_\theta(y))\|_1$$
其中 $\text{sg}(\cdot)$ 为 stop-gradient，$\bar{E}_\theta$ 为 EMA 编码器。

**是否 action-conditioned**：**否**。V-JEPA 不使用任何动作条件（action conditioning）。条件变量 $\Delta y$ 仅为被遮挡区域的时空位置信息，不包含动作信号。

**3D Multi-Block Masking 策略**：每个 clip 同时采样两种 mask（multi-mask prediction，用于摊销目标计算成本）：

| 掩码类型 | 采样块数 | 每块空间覆盖率 | 特点 |
|---------|---------|-------------|-----|
| Short-range mask | 8 块 | 15%（spatial scale=0.15） | 多小块联合 |
| Long-range mask | 2 块 | 70%（spatial scale=0.7） | 少量大块联合 |

共同设置：
- 宽高比在 (0.75, 1.5) 范围内随机选取
- 空间块沿**整个时间维度**延伸（temporal ratio=100%），防止时空冗余带来的信息泄漏
- 取多个块的**并集**作为最终掩码，平均掩码率约 **~90%**

掩码策略消融对比（ViT-L/16 on K710+SSv2）：

| 策略 | K400 | SSv2 | IN1K |
|------|------|------|------|
| random-tube[0.9] | 51.5 | 46.4 | 55.6 |
| causal multi-block[6] | 61.3 | 49.8 | 66.9 |
| causal multi-block[12] | 71.9 | 63.6 | 72.2 |
| **multi-block（本文）** | **72.9** | **67.4** | **72.8** |

### Dec()

**V-JEPA 刻意不设置 Decoder。** 设计动机：

- 像素空间预测需要消耗大量模型容量和计算来捕获所有低级细节（纹理、光照），而这些对下游语义任务无用。
- 特征空间预测允许模型**自动忽略不可预测或无关的像素级细节**，只保留语义上有意义的信息。
- 理论动机：在 L1 loss 下，最优 Predictor 输出目标的条件中位数（conditional median），encoder 必须学习捕获尽可能多的视频信息以最小化中位绝对偏差（MAD）。

**关于可视化 Decoder 的说明**：论文为定性分析，在第 6 节额外训练了一个**条件扩散模型 decoder**，将 V-JEPA 的特征预测解码回像素。但这个 decoder 与预训练完全分离，预训练时 encoder 和 predictor 保持冻结，decoder 仅用于可视化，不参与预训练。

### 关键设计亮点

1. **时间维度全延伸掩码（Temporal Full Masking）**：所有空间 mask block 沿整个时间维度延伸（temporal ratio=100%），防止模型利用时序冗余（从相邻帧推断被遮挡内容）走捷径，迫使学习真正的时序语义理解。

2. **Attentive Probing（下游评估）**：使用可学习的 cross-attention 层（12 头×12 维）+ 残差 + 2 层 MLP + LayerNorm 进行特征聚合，相比平均池化在 K400 上提升 +17.3 点，SSv2 提升 +16.1 点，是评估视频表示质量的重要方法论贡献。

3. **Multi-Mask Prediction 提升效率**：每个 clip 同时采样 2 种掩码（short-range + long-range），共享一次 y-encoder 前向传播，x-encoder 和 predictor 分别各做一次，在不增加计算量的前提下提升训练信号的多样性。

## 实验结果

### 预训练数据集：VideoMix2M

HowTo100M + Kinetics-400/600/700 (K710) + Something-Something-v2 (SSv2)，去除与验证集重叠后约 **200 万个视频**。训练 90,000 steps，处理样本约 270M（ViT-L/16, ViT-H/16）。

### 核心结果（Frozen Evaluation，Attentive Probing）

| 方法 | 架构 | 参数量 | K400 | SSv2 | AVA | IN1K |
|------|------|--------|------|------|-----|------|
| I-JEPA | ViT-H/16 | 630M | 79.7 | 50.0 | 19.8 | 84.4 |
| DINOv2 | ViT-g/14 | 1100M | 83.4 | 50.6 | 24.3 | 86.2 |
| VideoMAE | ViT-H/16 | 630M | 79.8 | 66.2 | 20.7 | 72.3 |
| VideoMAEv2 | ViT-g/14 | 1100M | 71.2 | 61.2 | 12.9 | 71.4 |
| MVD | ViT-L/16 | 200M | 79.4 | 66.5 | 19.7 | 73.3 |
| **V-JEPA** | **ViT-L/16** | **200M** | **80.8** | **69.5** | **25.6** | **74.8** |
| **V-JEPA** | **ViT-H/16** | **630M** | **82.0** | **71.4** | **25.8** | **75.9** |
| **V-JEPA** | **ViT-H/16₃₈₄** | **630M** | **81.9** | **72.2** | **25.0** | **77.4** |

### 与像素预测方法对比（ViT-L/16，Frozen + Fine-tuning）

| 方法 | 样本量 | 迭代数 | K400-frozen | SSv2-frozen | K400-ft | SSv2-ft |
|------|--------|--------|-------------|-------------|---------|---------|
| OmniMAE | 2400M | 1170K | 65.6 | 60.6 | 84.0 | 74.2 |
| VideoMAE | 410M | 400K | 77.8 | 65.5 | 85.4 | 74.3 |
| Hiera | 770M | 1500K | 75.5 | 64.2 | 87.3 | 75.1 |
| **V-JEPA** | **270M** | **90K** | **80.8** | **69.5** | **85.6** | **75.1** |

V-JEPA 以**少一个数量级的样本量和迭代数**超越所有像素预测基线。

### 低样本评估（Frozen，V-JEPA ViT-H/16₃₈₄）

| 标签比例 | K400 | SSv2 |
|---------|------|------|
| 5% | 68.2% | 54.0% |
| 10% | 72.8% | 59.3% |
| 50% | 80.6% | 67.9% |

对比：VideoMAEv2 ViT-g/14 在 5% 标签下仅 37.0% / 28.0%。

## 局限性

- **ImageNet 与图像模型仍有差距**：V-JEPA 在 ImageNet 上的表现（77.4-77.9%）仍落后于 DINOv2（86.2%）和 OpenCLIP（85.3%）等大规模图像预训练模型，差距超过 8 个百分点。论文将其归因于预训练数据的视觉多样性不足。
- **视频预训练数据集规模和多样性受限**：论文明确指出，"用于训练 V-JEPA 和其他视频模型的数据集过于受限，缺乏图像模型使用的互联网规模预训练数据的视觉多样性"，并呼吁未来构建多样化的公开视频数据集。
- **K400 上不及最强图像模型**：在 Kinetics-400（外观依赖型任务）上，V-JEPA ViT-H/16（82.0%）略低于 DINOv2 ViT-g/14（83.4%），因为 K400 许多类别可从外观线索直接推断，图像模型具有天然优势。
- **非生成模型，无法直接生成像素**：V-JEPA 在特征空间预测，不能直接生成可解释的像素级输出；论文中的可视化需要额外训练一个独立的条件扩散解码器，增加了分析成本。
- **不包含 action conditioning**：V-JEPA 是纯自监督的视觉表示学习方法，不能直接用于需要动作条件的机器人规划任务，这一扩展由 V-JEPA 2 完成。

## 与 JEPA 系列演进关系

V-JEPA 是 JEPA 系列的**第二步**，将 I-JEPA（图像域）的核心思想直接扩展到视频域：

- **继承自 I-JEPA**：embedding space 中的预测、EMA 目标编码器、stop-gradient 防坍塌、无 Decoder 的设计哲学、窄 Predictor 结构。
- **对 I-JEPA 的扩展**：将 2D 空间 patch 扩展为 3D 时空 tubelet；引入时间维度全延伸掩码策略；从图像理解扩展到视频运动理解；引入 Attentive Probing 评估协议。
- **与 I-JEPA 的共同点**：两者均**不使用 action conditioning**，都是纯自监督的视觉表示学习方法。
- **为 V-JEPA 2 奠基**：V-JEPA 2 在 V-JEPA 的基础上扩展至 1B 参数和 100 万小时视频，并在冻结的 V-JEPA 2 encoder 基础上引入 action-conditioned 世界模型（V-JEPA 2-AC），实现机器人零样本规划。
- **与 LeWM 的关系**：LeWM 与 V-JEPA 2-AC 共享 action-conditioned latent world model 的核心思路，但 LeWM 采用端到端训练（无冻结 encoder），并以 SIGReg 替代 EMA/stop-gradient 防坍塌。
