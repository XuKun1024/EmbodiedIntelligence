# Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

> **作者**：Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas  
> **机构**：Meta AI (FAIR)；McGill University；Mila, Quebec AI Institute；New York University  
> **年份/会议**：2023 / CVPR 2023  
> **arXiv**：arXiv:2301.08243

## 一句话总结

I-JEPA 是一种非生成式自监督图像学习方法，通过在抽象表示空间（而非像素空间）中预测被遮挡图像块的表示，无需任何手工数据增强即可学到同时具备高层语义与低层局部特征的视觉表示，且计算效率比同类方法高出一个数量级。

## 核心贡献

- **提出 I-JEPA 框架**：在 representation space 而非 pixel/token space 中做预测，无需 view augmentations，避免了手工设计偏置的引入。
- **Multi-Block Masking 策略**：预测多个具有足够大尺度的目标块（引导语义理解），同时使用空间分布广泛的单一上下文块（提供丰富上下文），是引导模型学习语义表示的核心设计。
- **超越像素重建方法的语义质量**：在 ImageNet 线性探测、半监督 1% ImageNet 等任务上显著超越 MAE、CAE、data2vec 等不依赖数据增强的方法，同时在低层任务（对象计数、深度预测）上超越 DINO、iBOT 等依赖手工增强的方法。
- **极高计算效率**：ViT-H/14 在 16 块 A100 GPU 上不到 72 小时（< 1200 GPU 小时）完成预训练，比 iBOT ViT-S/16 快超过 2.5 倍，比 MAE ViT-H/14 快超过 10 倍，收敛所需迭代数减少约 5 倍。
- **同时捕获高层语义与低层局部特征**：在 Clevr/Count（对象计数）和 Clevr/Dist（深度预测）上超越依赖手工增强的 DINO 和 iBOT，展示更广泛的任务适用性。

## 方法

### Enc()

**Context Encoder（上下文编码器）**：

- 架构：标准 Vision Transformer (ViT)，支持 ViT-B/16、ViT-L/16、ViT-H/14、ViT-H/16（448 分辨率）、ViT-G/16 等配置。
- 输入处理：只处理可见的上下文 patch（不处理被 mask 的 patch），类似 MAE 的高效稀疏编码策略，不使用 [CLS] token，以平均池化输出作为全局表示。
- 默认输入分辨率：224×224（部分实验用 448×448）。

**Target Encoder（目标编码器）**：

- 与 Context Encoder 结构完全相同，但权重不通过梯度更新，而是通过 Context Encoder 权重的指数移动平均（EMA）更新。
- EMA 动量初始值 0.996，线性增加到 1.0。
- 评估时取其输出的平均池化作为全局图像表示。

### Pred()

**预测目标**：在 embedding space 中预测目标块（target blocks）的 patch 级别表示向量。具体地，给定上下文编码器输出 $s_x$，对每个目标块 $B_i$，Predictor $g_\phi$ 接收 $s_x$ 和对应位置的 mask token $\{m_j\}_{j \in B_i}$，输出预测表示 $\hat{s}_y^{(i)}$，与 Target Encoder 输出做 L2 回归。

**Predictor 架构**：

- 轻量级（narrow）ViT 架构，embedding 维度固定为 384（形成瓶颈，encoder 的 embedding 维度通常更大，如 ViT-L 为 1024）。
- 层数：ViT-B/16 对应 6 层；ViT-L/16、ViT-H 对应 12 层；ViT-G/16 对应 16 层。
- Mask token：共享的可学习向量 + 位置编码，指示被遮挡位置。
- Predictor 对 M 个目标块分别运行 M 次，每次条件化在不同目标块的位置 token 上。

**是否 action-conditioned**：否。I-JEPA 不使用任何动作条件，条件变量仅为被遮挡区域的空间位置信息。

**Multi-Block Masking 策略（具体参数）**：

| 角色 | 参数 |
|------|------|
| 目标块数量 M | 4（可能有重叠） |
| 目标块 scale 范围 | (0.15, 0.2)（占图像面积比例） |
| 目标块长宽比范围 | (0.75, 1.5) |
| 上下文块数量 | 1 |
| 上下文块 scale 范围 | (0.85, 1.0)（大范围，空间分布广） |
| 上下文块处理 | 移除与任意目标块重叠的区域，确保预测任务非平凡 |

关键细节：掩码操作作用于 **Target Encoder 的输出端**（而非输入端）——消融实验显示输出端掩码（67.3% top-1）远优于输入端掩码（56.1%），确保目标表示具有高语义层次。

### Dec()

**I-JEPA 刻意不设置 Decoder。** 设计动机如下：

- 生成式方法（如 MAE）在像素/token 空间重建，需要处理大量与语义无关的低层细节（纹理、光照等），导致学到的表示语义层次较低，在线性探测等 off-the-shelf 评估中表现不佳。
- I-JEPA 在抽象 representation space 中做预测，Target Encoder 有能力消除不必要的像素级细节，从而引导模型学习更语义的特征。
- 论文用消融实验直接验证：在像素空间计算损失（Top-1 40.7%）vs. 在表示空间计算损失（Top-1 66.9%），差距巨大。

**关于可视化**：论文为定性分析训练了一个额外的 RCDM 扩散模型解码器，将 Predictor 输出解码回像素，但这与预训练完全分离，仅用于可视化，不参与预训练。

### 关键设计亮点

1. **表示空间预测 vs. 像素空间重建**：JEPA 的核心哲学，避免学习无关低层细节，引导模型学习语义表示。消融实验（像素空间 40.7% vs. 表示空间 66.9%）直接验证了这一设计的必要性。

2. **Predictor 宽度瓶颈**：窄 Predictor（384 维）在 ImageNet 1% 上优于宽 Predictor（1024 维），分别为 70.7% vs. 68.4%。瓶颈强迫 Predictor 学习更抽象的表示，而非简单记忆。

3. **Multi-Block Masking 的必要性**：消融实验显示 multi-block（54.2%）远优于 rasterized（15.5%）、block（20.2%）、random（17.6%）等掩码策略。多块目标 + 大范围上下文的组合是学到语义表示的关键。

## 实验结果

### ImageNet-1K 线性探测（不依赖手工增强的方法对比）

| 方法 | 架构 | Epochs | Top-1 |
|------|------|--------|-------|
| data2vec | ViT-L/16 | 1600 | 77.3 |
| MAE | ViT-H/14 | 1600 | 77.2 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| **I-JEPA** | **ViT-H/14** | **300** | **79.3** |
| **I-JEPA** | **ViT-H/16（448）** | **300** | **81.1** |
| iBOT（有增强） | ViT-L/16 | — | 81.0 |

### ImageNet-1% 半监督评估

| 方法 | 架构 | Top-1 |
|------|------|-------|
| MAE | ViT-H/14 | 71.5 |
| **I-JEPA** | **ViT-H/14** | **73.3** |
| **I-JEPA** | **ViT-H/16（448）** | **77.3** |
| MSN（有增强） | ViT-B/4 | 75.7 |

### 迁移学习线性探测

| 方法 | 架构 | CIFAR100 | Places205 | iNat18 |
|------|------|----------|-----------|--------|
| MAE | ViT-H/14 | 77.3 | 55.0 | 32.9 |
| **I-JEPA** | **ViT-H/14** | **87.5** | **58.4** | **47.6** |
| DINO（有增强） | ViT-B/8 | 84.9 | 57.9 | 55.9 |
| iBOT（有增强） | ViT-L/16 | 88.3 | 60.4 | 57.3 |

I-JEPA 在 CIFAR100 和 Places205 上超越 DINO，但在 iNat18（细粒度分类）上仍落后于有增强的方法。

### 低层任务（Clevr 数据集）

| 方法 | 架构 | Clevr/Count | Clevr/Dist |
|------|------|-------------|------------|
| MAE | ViT-H/14 | 90.5 | 72.4 |
| **I-JEPA** | **ViT-H/14** | **86.7** | **72.4** |
| DINO（有增强） | ViT-B/8 | 86.6 | 53.4 |
| iBOT（有增强） | ViT-L/16 | 85.7 | 62.8 |

I-JEPA 在深度预测上以大幅优势超越 DINO（72.4 vs. 53.4）和 iBOT（72.4 vs. 62.8）。

### 计算效率

- ViT-H/14 预训练：< 1200 GPU 小时（16 块 A100，< 72 小时）
- 比 iBOT ViT-S/16 快超过 2.5 倍，比 MAE ViT-H/14 快超过 10 倍
- 收敛所需迭代数减少约 5 倍

## 局限性

- **与视图不变性方法在细粒度任务上仍有差距**：在 iNat18（细粒度分类）上，I-JEPA ViT-H/14（47.6%）仍明显落后于 iBOT ViT-L/16（57.3%）和 DINO ViT-B/8（55.9%），说明手工增强带来的视图不变性在细粒度语义任务上仍有价值。
- **更大 patch 尺寸不利于低层局部任务**：ViT-G/16 使用更大输入 patch，在 Clevr/Count、Clevr/Dist 上相比 ViT-H/14 没有提升甚至略有下降，说明 patch 大小与任务粒度之间存在权衡。
- **不直接适用于像素级生成任务**：由于刻意避免像素空间预测，I-JEPA 不适合图像生成、图像修复等需要精细像素输出的任务，这是设计上的取舍。
- **Predictor 可视化需要外部解码器**：论文用 RCDM 扩散模型解码器来可视化 Predictor 的输出，这并非端到端，是一种后验分析工具。

## 与 JEPA 系列演进关系

I-JEPA 是 JEPA 系列的**起点**，由 Yann LeCun 在"A Path Towards Autonomous Machine Intelligence"（2022）中提出的 JEPA 框架在图像域的第一个具体实现。

- **I-JEPA → V-JEPA**：V-JEPA 将 I-JEPA 的核心思想扩展到视频域，用 3D 时空 patch（tubelet）替代 2D 空间 patch，引入时空多块掩码策略，将预测对象从图像 patch 表示扩展为视频时空区域表示。两者均不使用 action conditioning。
- **I-JEPA → V-JEPA 2**：V-JEPA 2 在更大规模（1B 参数，100 万小时视频）上扩展 V-JEPA，并在此基础上引入 action-conditioned 世界模型（V-JEPA 2-AC），向机器人应用延伸。
- **I-JEPA → LeWM**：LeWM 将 JEPA 的 embedding space 预测哲学直接应用于 action-conditioned latent world model，是 JEPA 框架在端到端机器人规划上的直接延伸，但不依赖 EMA/stop-gradient 等启发式手段。

I-JEPA 确立了 JEPA 系列的三个核心原则：①在 embedding space 中预测；②无 Decoder；③EMA 目标编码器防止表示坍塌。后续工作均在此基础上扩展。
