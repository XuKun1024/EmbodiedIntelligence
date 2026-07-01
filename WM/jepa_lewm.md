# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

> **作者**：Lucas Maes\*, Quentin Le Lidec\*, Damien Scieur, Yann LeCun, Randall Balestriero（\* 共同第一作者）  
> **机构**：Mila & Université de Montréal；New York University；Samsung SAIL；Brown University  
> **年份/会议**：2026 / 预印本（arXiv）  
> **arXiv**：arXiv:2603.19312v3（最新版本 2026 年 6 月 3 日）

## 一句话总结

LeWM 是首个无需任何训练启发式手段（stop-gradient、EMA、预训练编码器）即可从原始像素端到端稳定训练的 action-conditioned JEPA 世界模型，以仅 15M 参数和单一超参数（λ），在多种机器人操控和导航任务上超越现有端到端 JEPA 方法，并以约 48 倍的速度优势超越基于 foundation model 的规划方法。

## 核心贡献

- **首个稳定的端到端 JEPA**：LeWM 是第一个无需任何训练启发式手段（stop-gradient、EMA、预训练编码器）即可从原始像素端到端稳定训练的 JEPA 方法，梯度完全端到端传播。
- **极简训练目标，超参数从 6 个缩减到 1 个**：训练目标仅包含两项：next-embedding 预测损失（MSE）+ SIGReg 正则化（强制 latent embeddings 服从高斯分布）。相比唯一的端到端竞争对手 PLDM 的 7 项损失（6 个可调权重），LeWM 只有 1 个实质性超参数 λ，可用二分搜索以 O(log n) 复杂度高效调优。
- **高效轻量：15M 参数，单 GPU 数小时可训完，规划速度比基于 foundation model 的方法快 48×**：编码观测所用 token 数比 DINO-WM 少约 200×，LeWM 完整规划仅需约 0.98 秒，而 DINO-WM 需要约 47 秒。
- **竞争性控制性能，跨多种 2D/3D 任务**：在 Push-T（2D 操作）、OGBench-Cube（3D 操作）、Reacher（2D 运动规划）、TwoRoom（2D 导航）等多种连续动作空间任务上，LeWM 超越了现有端到端 JEPA 方法（PLDM），并在固定计算量下显著超越 DINO-WM。
- **Latent 空间具备物理结构理解能力**：通过 probing 实验验证 latent 表征编码了物理量（位置、角度等），并通过违反期望（Violation-of-Expectation, VoE）框架验证模型能可靠检测物理上不合理的事件（如物体瞬移）。

## 方法

### Enc()

**架构**：Vision Transformer (ViT)，使用 **ViT-Tiny** 配置（默认）：
- 约 **5M 参数**
- Patch size = 14
- 12 层，3 个注意力头，隐藏维度 192
- 输入：224×224 RGB 图像，每帧独立编码

**输出**：取最后一层的 **[CLS] token embedding**，后接一个 **1 层 MLP + Batch Normalization** 投影层（projection head）。注意：最后一层 ViT 有 Layer Normalization，因此需要 BN 投影层才能让 SIGReg 正则化有效优化。

**输出维度**：192 维 latent embedding（默认）。

**关键区别**：LeWM 的 encoder 是**端到端可训练的**，不冻结，不使用 EMA 目标编码器，不使用 stop-gradient。这是与 I-JEPA、V-JEPA、V-JEPA 2 的根本区别。

### Pred()

**预测目标**：在 latent 空间中预测**下一时刻的 frame embedding**（next-embedding prediction）——给定当前 latent 状态 $z_t$ 和动作 $a_t$，预测 $\hat{z}_{t+1}$。不预测像素，不重建图像，完全在 compact latent 空间操作。

**架构**：ViT-Small backbone（约 10M 参数），6 层 Transformer，16 个注意力头，10% dropout，后接与 encoder 相同结构的 projector 网络（1 层 MLP + BN）。

**Mask 策略（时序因果掩码）**：使用 **temporal causal masking**（时序因果掩码），防止 predictor 看到未来的 embedding。Predictor 以 N 帧历史的 frame representations 作为输入，自回归地预测下一帧 representation（history length N：Push-T 和 OGBench-Cube 为 3，TwoRoom 为 1）。

**是否 action-conditioned**：**是**。LeWM 是 action-conditioned 世界模型，这是其与 I-JEPA、V-JEPA 的根本区别。

**Action Conditioning 机制**：通过 **Adaptive Layer Normalization（AdaLN）** 将动作条件注入 predictor 的每一层。AdaLN 参数初始化为零，确保动作条件对 predictor 训练的影响是渐进的（stabilize training）。这一技术借鉴自 DiT（Scalable Diffusion Models with Transformers）。

**规划时的自回归 rollout**：
$$\hat{z}_{t+1} = \text{pred}_\phi(\hat{z}_t, a_t), \quad \hat{z}_1 = \text{enc}_\theta(o_1)$$

**完整训练目标**：
$$\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z)$$

其中 $\mathcal{L}_{\text{pred}} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$（teacher-forcing MSE），$\lambda = 0.1$（默认）。

**其他工程细节**：Frame skip = 5（5 个连续动作合并为一个 action block）；batch size = 128；sub-trajectory 长度 = 4 帧。

### Dec()

**训练时：无 Decoder**（这是 LeWM 的核心设计哲学，无重建损失，完全在 latent space 操作）。

**可视化用途：有一个事后训练的轻量 Decoder（仅用于诊断）**：将 192 维 [CLS] token embedding 解码回 224×224 图像。架构：轻量 Transformer decoder，[CLS] 作为 key/value，196 个可学习 query tokens（对应 196 个 patch）通过 cross-attention 与全局表征交互，最终线性投影到 16×16×3 像素 patch。

**消融实验验证无 Decoder 的必要性**：加入重建损失反而会降低控制性能（Push-T: 96% → 86%），说明 JEPA 训练目标已经足够捕获规划所需信息，重建损失鼓励编码与控制无关的视觉细节。

### 关键设计亮点

1. **SIGReg（Sketched-Isotropic-Gaussian Regularizer）**：核心反坍塌机制，强制 latent embeddings 分布趋近各向同性高斯分布 $\mathcal{N}(0, I)$。利用 **Cramér-Wold 定理**：匹配所有 1D 边缘分布等价于匹配联合分布。具体做法：将 embeddings $Z \in \mathbb{R}^{N \times B \times d}$ 投影到 $M$ 个随机单位方向 $u^{(m)} \in S^{d-1}$，对每个 1D 投影 $h^{(m)} = Zu^{(m)}$ 计算 **Epps-Pulley 正态检验统计量**：
$$\text{SIGReg}(Z) = \frac{1}{M} \sum_{m=1}^{M} T(h^{(m)})$$
默认使用 M=1024 个随机投影，λ=0.1。这是与 PLDM 等启发式方法的本质区别——SIGReg 具有**可证明的反坍塌保证**。

2. **端到端训练无启发式手段**：无 stop-gradient，无 EMA，无预训练编码器，梯度完全端到端传播。这使得 encoder 可以同时为预测任务和规划任务优化表示，而非被冻结在预训练状态。

3. **极致轻量与高效**：仅 15M 参数（ViT-Tiny encoder + ViT-Small predictor），编码每帧仅用 1 个 [CLS] token（vs. DINO-WM 的 196 个 patch tokens），规划速度约 48 倍于 DINO-WM。

## 实验结果

### 评估环境

| 环境 | 类型 | 数据量 |
|------|------|--------|
| Push-T | 2D 操作（推 T 形块到目标位置） | 20,000 episodes |
| OGBench-Cube | 3D 操作（机械臂拾取放置方块） | 10,000 episodes |
| Reacher | 2D 运动规划（双关节臂到达目标） | 10,000 episodes |
| TwoRoom | 2D 导航（穿越门洞到达目标） | 10,000 episodes |

### 规划性能（Success Rate %）

**Push-T（核心基准）**：

| 方法 | Success Rate |
|------|-------------|
| **LeWM（ours）** | **96%** |
| DINO-WM+proprioception | 92% |
| DINO-WM | 74% |
| PLDM | 78% |
| GCBC | 75% |
| GCIQL | 20% |
| GCIVL | 33% |
| Random | 2% |

**Reacher**：LeWM **86%** vs. PLDM 78% vs. DINO-WM 79%

**OGBench-Cube**：DINO-WM **86%** > LeWM **74%** > PLDM 65%（LeWM 在此任务上不及 DINO-WM）

**TwoRoom**：DINO-WM/PLDM/GCBC 97-100% > LeWM **87%**（LeWM 在低多样性环境中表现较弱）

### 规划速度（Fixed FLOPs 对比）

| 方法 | 完整规划时间 | 固定计算量下 Push-T | 固定计算量下 OGBench-Cube |
|------|------------|-------------------|------------------------|
| LeWM | **0.98 秒** | **90%** | **74%** |
| DINO-WM | **47 秒** | 13% | 48% |

速度提升：**约 48×**。

### 训练稳定性（3 个随机种子，Push-T）

| 方法 | Success Rate |
|------|-------------|
| **LeWM** | **96.0 ± 2.83** |
| DINO-WM | 92.0 ± 1.63 |
| PLDM | 78.0 ± 5.0（方差最大） |

### 物理量 Probing（Push-T，Linear/MLP 探针）

| 物理量 | Linear r | MLP r |
|--------|----------|-------|
| Agent Location | 0.974 | 0.998 |
| Block Location | 0.986 | 0.999 |
| Block Angle | 0.902 | 0.990 |

LeWM 在所有指标上均超越 PLDM，与 DINO-WM（在 124M 图像上预训练）保持竞争力。

### Violation-of-Expectation（VoE）评估

- 物理扰动（物体瞬移）：LeWM 在所有三个环境中均产生显著 surprise 峰值（paired t-test, p < 0.01）。
- 视觉扰动（颜色突变）：surprise 增加较弱且不显著。
- 结论：LeWM 对物理连续性比视觉外观更敏感，与物理世界模型的期望行为一致。

## 局限性

- **规划视野受限（Short Horizon Planning）**：规划仍局限于短视野，需要层次化世界模型（hierarchical world modeling）来支持长视野推理。
- **对离线数据集覆盖度的依赖**：方法依赖具有充分覆盖度的离线数据集；在低多样性、低内在维度的环境（如 TwoRoom）中，SIGReg 难以在高维 latent 空间中有效匹配高斯先验，导致 latent 表征结构性较差，规划性能下降。
- **数据效率与泛化**：在大规模多样化视频数据集上预训练可以提供更强的先验并减少对特定领域数据的需求（暗示当前方法在数据效率上仍有提升空间）。
- **对动作标签的依赖**：当前方法需要动作标签监督；可通过逆动力学建模（inverse dynamics modeling）来缓解这一依赖。

## 与 JEPA 系列演进关系

LeWM 明确定位为 **JEPA 框架的直接延伸**，特别是 action-conditioned latent world model 变体。

**JEPA 谱系中的位置**：

| 特性 | I-JEPA | V-JEPA | V-JEPA 2 | V-JEPA 2-AC | LeWM |
|------|--------|--------|---------|------------|------|
| 域 | 图像 | 视频 | 视频 | 视频+机器人 | 图像（控制） |
| Action-conditioned | 否 | 否 | 否 | 是 | 是 |
| 端到端训练 | 是（EMA+SG） | 是（EMA+SG） | 是（EMA+SG） | 否（冻结 encoder） | 是（无 EMA/SG） |
| 反坍塌机制 | EMA + SG（启发式） | EMA + SG（启发式） | EMA + SG（启发式） | 预训练 encoder | SIGReg（可证明） |
| 超参数数量 | 多 | 多 | 多 | 少 | 1 个（λ） |
| 是否有重建损失 | 否 | 否 | 否 | 否 | 否 |
| 规划方式 | 不适用 | 不适用 | 不适用 | CEM in latent space | MPC in latent space |

**LeWM 对 JEPA 框架的关键推进**：

LeWM 解决了 action-conditioned JEPA 的核心挑战——**表征坍塌（representation collapse）**，提供了一个有理论保证（Cramér-Wold 定理）、实践简单（单一超参数）、训练稳定（无启发式手段）的解决方案。

- **与 I-JEPA/V-JEPA 的共同点**：在 embedding space 中预测，无 Decoder，轻量 Predictor 结构。
- **与 I-JEPA/V-JEPA 的区别**：LeWM 是 action-conditioned 的，而 I-JEPA/V-JEPA 是纯自监督的；LeWM 无 EMA/stop-gradient，而 I-JEPA/V-JEPA 依赖这些启发式手段防止坍塌。
- **与 V-JEPA 2-AC 的区别**：V-JEPA 2-AC 冻结预训练 encoder（依赖大规模预训练），LeWM 端到端训练 encoder（从随机初始化开始），规模更小但更轻量。
- **与 PLDM 的区别**：PLDM 是最接近的竞争对手（同为端到端 JEPA），但使用 VICReg 衍生的 7 项损失（6 个可调权重），训练不稳定；LeWM 以 SIGReg 替代，大幅简化且更稳定。

SIGReg 来自同组的 LeJEPA 工作（Balestriero & LeCun 2025，arXiv:2511.08544），LeWM 将其成功应用于 action-conditioned world model 的端到端训练场景，是 JEPA 系列在具身智能方向的重要推进。
