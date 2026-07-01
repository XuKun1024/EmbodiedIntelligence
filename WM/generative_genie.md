# Genie: Generative Interactive Environments

> **作者**：Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge (Jimmy) Shi, Tim Rocktäschel 等（共 25 人，均等贡献核心作者 6 人）  **机构**：Google DeepMind（全部作者）；Jeff Clune 同时兼属 University of British Columbia  **年份/会议**：2024 / ICML 2024  **arXiv**：arXiv:2402.15391

## 一句话总结

Genie 是第一个完全从无标注互联网视频中无监督训练的 110 亿参数基础世界模型，通过学习隐式动作空间（Latent Action Model）实现帧级可控交互，并能根据文本、草图或照片等多种 prompt 生成可交互的虚拟环境。

## 核心贡献

- 提出首个**无监督基础世界模型**：训练时无需任何动作标签，仅依赖无标注互联网视频即可学习可控的交互式环境生成
- 提出 **Latent Action Model（LAM）**：从视频帧对中无监督学习离散隐式动作空间（|A|=8），使模型在推理时可接受用户动作输入并产生对应的下一帧
- 实现 **110 亿参数**的大规模世界模型，是发表时规模最大的世界模型，并展示出一致的 scaling law
- 支持多模态 prompt 泛化：用户可用文本（经 Imagen2 生成图像后）、AI 生成图像、手绘草图或真实照片作为初始帧，生成对应风格的可交互世界
- 验证了 LAM 推断的隐式动作可用于对未见视频打标签，以少至 **200 条专家样本**的行为克隆（BC）达到 Oracle 水平的策略性能

## 方法

Genie 由三个共享 ST-Transformer 骨干网络的组件构成：**ST-ViViT 视频 Tokenizer**、**Latent Action Model（LAM）** 和 **Dynamics Model**。

---

### Enc()（ST-ViViT 视频 Tokenizer）

**输入与编码方式：**

输入为 T 帧视频 x_{1:T} ∈ ℝ^{T×H×W×C}，通过 ST-ViViT（Spatiotemporal Vision Transformer with VQ-VAE）编码为离散 token z_{1:T} ∈ 𝕀^{T×D}。每个 token z_t 因果地编码了所有先前帧的信息（causal 结构）。

**架构（ST-Transformer 骨干）：**

每个 ST-Transformer block 包含：
- **Spatial attention**：在单个时间步内的 H×W token 上做注意力（复杂度随帧数线性增长，而非平方级）
- **Temporal attention（因果）**：在 T×1×1 的时间轴上做因果注意力
- **单个 FFW 层**（刻意省略 spatial FFW，以控制计算量）

**Encoder 超参数（Platformers 域）：**

| 参数 | Encoder | Decoder |
|------|---------|---------|
| num_layers | 12 | 20 |
| d_model | 512 | 1024 |
| num_heads | 8 | 16 |
| Codebook 大小 | 1024 codes | — |
| patch_size | 4 | — |
| latent_dim | 32 | — |

**总参数量：** 200M

**设计动机：** 相比 C-ViViT（Phenaki 使用的全时空注意力，内存占用 1.6GB），ST-ViViT 的分离式时空注意力仅需 0.9GB 内存，且在 FVD 和 ΔPSNR 上均优于 ViT（纯空间）和 C-ViViT，是效率与质量的最优平衡点。

---

### Pred()（Dynamics Model + Latent Action Model）

**Latent Action Model（LAM）— 隐式动作学习：**

- **目的：** 从连续帧对中无监督学习离散隐式动作，无需任何人工动作标签
- **输入：** 原始像素帧（非 token，消融实验验证像素输入优于 token 输入）
- **Encoder：** 接收 x_{1:t} 和下一帧 x_{t+1}，输出连续隐式动作 ã_{1:t}
- **VQ 瓶颈：** 量化为离散 codebook，动作空间大小 |A|=8（刻意保持小值以确保人类可操控性）
- **Decoder（仅训练时使用）：** 接收历史帧 + 隐式动作，预测 x̂_{t+1}；推理时丢弃
- **总参数量：** 300M

**Dynamics Model — 在 latent 空间预测下一帧：**

- **预测空间：** 离散 token 空间（z 空间）
- **是否 action-conditioned：** 是，以 LAM 推断的隐式动作 ã_{1:t-1} 作为条件
- **架构：** Decoder-only MaskGIT Transformer（基于 ST-Transformer 骨干）
- **输入：** z_{1:t-1}（视频 token）+ ã_{1:t-1}（隐式动作 embedding，以**加法 embedding** 方式注入，而非拼接）
- **预测目标：** 下一帧的离散 token ẑ_t
- **训练损失：** ẑ_{2:T} 与 z_{2:T} 之间的交叉熵
- **Masking：** 对输入 token z_{2:T-1} 以 Bernoulli 概率（∈ [0.5, 1.0]）随机掩码
- **推理：** 每帧 25 步 MaskGIT 采样，温度=2.0，随机采样
- **训练稳定性：** bfloat16 + QK normalization

**Dynamics Model 超参数（最终版本）：**

| 参数 | 数值 |
|------|------|
| 参数量 | 10.1B |
| num_layers | 48 |
| num_heads | 36 |
| d_model | 5120 |
| k/q_size | 128 |
| batch_size | 512 |
| 训练步数 | 125k |
| 训练硬件 | 256 张 TPU-v5p |
| FLOPs | 6.6×10²² |
| 训练 token 数 | 942B |

---

### Dec()（ST-ViViT Decoder）

**有无 Decoder：** 有，ST-ViViT 的 Decoder 部分（20 层，d_model=1024，num_heads=16）。

**输出：** 将预测的离散 token ẑ_t 解码为像素帧 x̂_t。

**训练/推理中的角色：**
- 训练时：作为 VQ-VAE 的一部分，与 Encoder 联合训练（重建损失 + codebook 对齐损失）；
- 推理时：Dynamics Model 预测出下一帧 token 后，由 Decoder 渲染为可视化帧，供用户交互和下一步输入。

**Tokenizer 训练配置：**

| 参数 | 数值 |
|------|------|
| 训练步数 | 300k |
| 优化器 | AdamW + cosine decay |
| 最大学习率 | 3e-4 |
| β₁/β₂ | 0.9/0.9 |
| weight_decay | 1e-4 |
| warmup | 10k 步 |
| batch=64 时 PSNR | 35.7 |
| batch=384 时 PSNR | 36.5 |

---

### 关键设计亮点

1. **LAM 的无监督动作发现（|A|=8 的离散动作空间）**：通过 VQ 瓶颈将视频帧间变化压缩为仅 8 种离散动作，既保证了人类可操控性（8 个按键），又避免了对任何动作标注的依赖。消融实验表明，使用原始像素作为 LAM 输入（而非 token）在可控性指标 ΔPSNR 上有显著优势（1.91 vs 1.33 on Platformers）。

2. **动作以加法 Embedding 注入（非拼接）**：Dynamics Model 中隐式动作 embedding 以加法方式叠加到 token embedding 上，而非像 IRIS 等工作那样将动作 token 拼接到序列中。这一设计更简洁且效果更好。

3. **ST-Transformer 的线性时空复杂度**：通过将空间注意力和时间注意力分离（而非全时空联合注意力），复杂度对帧数呈线性而非平方级增长，使得 110 亿参数规模的训练成为可能，且内存占用（0.9GB）远低于 C-ViViT（1.6GB）。

## 实验结果

### 数据集

**Platformers 数据集：**
- 初始收集：5500 万个 16 秒游戏视频片段（10 FPS，160×90 分辨率，约 24.4 万小时）
- 数据筛选：人工标注约 1 万个视频（约 10 小时人工成本），训练 1100 万参数 ResNet18 分类器进行质量过滤
- 最终数据集：**680 万个片段，约 3 万小时**
- 筛选效果（580M 参数模型）：FVD 从 61.4（5500 万原始视频）降至 54.8（680 万筛选后视频）

**Robotics 数据集：**
- RT-1 数据集（约 13 万机器人演示）+ 仿真数据 + 20.9 万真实机器人 episodes
- 动作标签刻意不使用，视为纯视频
- Robotics 模型：2.5B 参数，FVD=82.7

### Tokenizer 架构消融（Table，固定参数量，patch_size=10）

| 架构 | 参数量 | 内存 | FVD↓ | ΔPSNR↑ |
|------|--------|------|------|--------|
| ViT（纯空间） | 230M | 0.3GB | 114.5 | 1.39 |
| C-ViViT（Phenaki） | 225M | 1.6GB | 272.7 | 1.37 |
| **ST-ViViT（Genie）** | **205M** | **0.9GB** | **81.4** | **1.66** |

### LAM 输入消融（像素 vs. Token）

| 模型 | 数据集 | 参数量 | FVD↓ | ΔPSNR↑ |
|------|--------|--------|------|--------|
| Token 输入 | Platformers | 2.3B | **38.8** | 1.33 |
| 像素输入（Genie） | Platformers | 2.5B | 40.1 | **1.91** |
| Token 输入 | Robotics | 1B | 257.8 | 1.65 |
| 像素输入（Genie） | Robotics | 1B | **136.4** | **2.07** |

像素输入在可控性（ΔPSNR）上在两个域均优于 token 输入；token 输入在 Platformers 的 FVD 上略胜。

### Scaling Law（固定 batch=256，200k 步，750B token）

| 参数量 | num_layers | num_heads | d_model | 硬件 | FLOPs |
|--------|-----------|-----------|---------|------|-------|
| 41M | 18 | 8 | 512 | 64 TPUv2 | 2.05×10²⁰ |
| 96M | 16 | 16 | 768 | 64 TPUv2 | 3.58×10²⁰ |
| 192M | 20 | 18 | 1024 | 64 TPUv2 | 6.4×10²⁰ |
| 404M | 21 | 12 | 1536 | 64 TPUv2 | 1.2×10²¹ |
| 811M | 20 | 20 | 2048 | 128 TPUv3 | 2.2×10²¹ |
| 1.6B | 28 | 22 | 2560 | 128 TPUv3 | 4.04×10²¹ |
| 2.7B | 36 | 22 | 3072 | 256 TPUv3 | 6.91×10²¹ |

每次规模提升均带来训练损失的一致下降，验证了 scaling law 的存在。

### 行为克隆（CoinRun）

使用冻结 LAM 对专家视频推断隐式动作，训练策略 π(aₜ|xₜ)，再用少量标注样本将隐式动作映射为真实动作：
- **结论：** 仅需 **200 条专家样本**，LAM-based 策略即可达到 Oracle BC 的性能水平，且模型从未在预训练中见过 CoinRun 游戏。

### 可控性指标 ΔPSNR

ΔPSNR 是论文提出的新指标：ΔₜPSNR = PSNR(xₜ, x̂ₜ) − PSNR(xₜ, x̂ₜ′)，其中 x̂ₜ 使用真实推断动作，x̂ₜ′ 使用随机采样动作。ΔPSNR 越高表示模型越可控（在 t=4 时刻报告）。

## 局限性

1. **幻觉问题**：继承自自回归 Transformer 的通病，模型可能生成"不真实的未来"（hallucinate unrealistic futures）
2. **短期记忆**：上下文仅限 16 帧，难以在长时间跨度内保持环境一致性
3. **生成速度慢**：当前约 1 FPS，"需要未来技术进步才能实现适合交互的帧率"
4. **模型和数据未公开**：模型权重、训练数据及数据样本均未对外发布

## 与同范式其他工作的关系

**vs. GameNGen（Valevski et al., 2024）：**
GameNGen 使用显式动作条件（真实键盘输入）和扩散模型（SD v1.4），实现 20 FPS 实时运行；Genie 无监督学习隐式动作（无需标签），用 MaskGIT Transformer，当前约 1 FPS。GameNGen 更侧重实时性和逼真度，Genie 更侧重从无标注数据中学习可控性和泛化能力。

**vs. GAIA-1（Hu et al., 2023）和 UniSim（Yang et al., 2023）：**
两者均需要视频 + 文本 + 动作标签进行训练，实现帧级可控；Genie 无需任何动作标签，是唯一在无监督条件下实现帧级可控的模型。

**vs. Phenaki / C-ViViT（Villegas et al., 2023）：**
Phenaki 支持文本条件视频生成，但无帧级交互控制，且使用全时空注意力（二次复杂度，内存开销大）；Genie 的 ST-ViViT 用分离式时空注意力解决了这一问题，同时增加了帧级可控性。

**vs. VPT（Baker et al., 2022）：**
VPT 需要昂贵的人工动作标注来学习动作空间；Genie 的 LAM 完全无监督，大幅降低了数据采集成本。

**vs. IRIS / TWMN（Micheli et al., 2023）和 DreamerV2：**
这些模型均需要动作标签，且 IRIS 将动作 token 拼接至序列（而 Genie 使用加法 embedding）；Genie 在无动作标签的前提下达到可比的可控性。

**vs. Sora / 视频生成模型：**
Sora 等模型为离线、非交互式视频生成；Genie 将视频生成范式推进到可交互的世界模型阶段，是从"视频生成"到"可交互虚拟环境"的关键一步。项目页面还演示了 Imagen2 → Genie 的链式调用（文本 → 图像 → 可交互世界）。
