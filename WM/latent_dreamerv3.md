# Mastering Diverse Domains through World Models (DreamerV3)

> **作者**：Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap
> **机构**：Google DeepMind; University of Toronto
> **年份/会议**：2023（arXiv 预印本，v2 发布于 2024年4月）
> **arXiv**：arXiv:2301.04104

## 一句话总结

DreamerV3 通过一套基于归一化、平衡和变换的鲁棒性技术，实现了单一固定超参数集跨越 150+ 个多样化任务（Atari、DMControl、Minecraft、ProcGen、DMLab 等）的世界模型训练，并首次在无人类数据的条件下在 Minecraft 中自主收集钻石。

## 核心贡献

- **固定超参数跨域通用**：单一 DreamerV3 agent 使用完全相同的超参数，在 8 个基准（150+ 个任务）上均超越专门调优的基线，无需针对不同域重新调参。
- **首个无人类数据在 Minecraft 收集钻石的算法**：在稀疏奖励（12 个里程碑，每个 +1）、长时序（最长 36000 步/episode）、开放世界环境中，不依赖课程学习或专家演示，直接从零学习，100M 步内所有 10 个 seed 均发现钻石。
- **可预测的缩放性质**：模型规模从 12M 到 400M 参数，性能单调提升；更大模型还需要更少的环境交互步数；重放比（replay ratio）提升也带来可预测的性能增益。
- **一套鲁棒性技术**：symlog 变换、KL balancing + free bits、symexp twohot 损失、百分位回报归一化、unimix 分布、输出权重零初始化，系统性地解决了跨域训练中的尺度不一致、梯度爆炸、稀疏奖励等问题。

## 方法

### Enc()

输入为图像（CNN 编码）或低维向量（MLP 编码）。编码器将观测 x_t 映射到随机离散潜在表示 z_t：

```
z_t ~ q_φ(z_t | h_t, x_t)    ← 后验编码器（表示模型）
```

**图像输入**：使用多层 CNN（通道数随模型规模缩放，如 200M 模型为 64 通道）。
**向量输入**：使用 MLP，输入先经 **symlog 变换**处理：

```
symlog(x) = sign(x) · ln(|x| + 1)
```

symlog 压缩大正/负值的幅度，对原点附近近似恒等变换，防止大输入导致的梯度爆炸，同时支持负值（不同于 log）。

**潜在表示**：离散类别分布（categorical），使用 straight-through 梯度。表示向量为 32×32 的离散码（200M 模型），采样后拼接确定性状态 h_t 形成完整模型状态 s_t = {h_t, z_t}。

**Unimix（均匀混合）**：编码器分布为 99% 神经网络 softmax + 1% 均匀分布的混合，防止零概率和无穷 log 概率，确保 KL 损失行为良好。

### Pred()

RSSM（Recurrent State Space Model）包含两个预测器，构成完整的序列模型：

```
序列模型（确定性）：  h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})     ← Block GRU（8 块）
动力学预测器（随机）：ẑ_t ~ p_φ(ẑ_t | h_t)                     ← 先验，不依赖观测
```

- **Block GRU**：将 GRU 分为 8 个并行块，提升计算效率和表达能力（相比 DreamerV2 的标准 GRU）。
- **动力学预测器**在潜空间中预测下一个随机状态，用于想象轨迹展开（无需观测输入）。
- **归一化**：使用 RMSNorm；**激活函数**：SiLU；**优化器**：AGC（自适应梯度裁剪，30% L2 范数阈值）+ LaProp（RMSProp 后接 momentum，ε=10⁻²⁰）。

**想象轨迹**：从重放经验的模型状态出发，按动力学预测器 + 策略前向展开 T=16 步，完全在潜空间中进行。

### Dec()

**观测解码器**：

```
x̂_t ~ p_φ(x̂_t | h_t, z_t)    ← 图像用转置 CNN，向量用 MLP
```

- **图像解码器**：转置 CNN，输出高斯分布（MSE 损失）。
- **向量解码器**：MLP，目标经 symlog 变换后用**平方损失**（symlog 平方损失），输出经 symexp 逆变换还原：

  ```
  symexp(x) = sign(x) · (exp(|x|) - 1)
  ```

- **训练时**：解码器提供主要的表征学习信号（预测损失 L_pred），是 DreamerV3 与 TD-MPC2（无解码器）的根本区别。消融实验显示去掉重建梯度会导致性能大幅下降。
- **规划/行为学习时**：actor-critic 直接在潜空间 s_t 上操作，不使用解码器。

**继续预测器（Continue Predictor）**：

```
ĉ_t ~ p_φ(ĉ_t | h_t, z_t)    ← 二元逻辑回归，预测 episode 是否继续
```

用于处理有终止条件的任务，折扣因子 γ=0.997（对应折扣视野 333 步）。

### Reward / Value 建模

**奖励预测器**：MLP，以 (h_t, z_t) 为输入，使用 **symexp twohot 损失**：

```
分箱：B = symexp(linspace(-20, +20))   ← 101 个指数间隔的 bin
输出：ŷ = softmax(f(x))ᵀ · B
损失：L = -twohot(y)ᵀ · log(softmax(f(x, θ)))
```

- Twohot 编码：将连续目标值 y 分配给最近两个 bin（权重之和为 1），其余 bin 权重为 0。
- 关键优势：梯度大小与目标绝对值解耦，避免大奖励导致梯度爆炸，使固定超参数跨域成为可能。
- **输出权重初始化为零**：防止随机初始化的大预测奖励延迟早期学习。

**Critic（价值函数）**：

- 预测**回报的分布**（而非期望值），同样使用 symexp twohot 损失。
- 使用 bootstrapped λ-return（λ=0.95，视野 T=16）。
- **双重 Critic 损失**：
  - 想象轨迹上的价值损失（β_val=1）
  - 重放缓冲区轨迹上的价值损失（β_repval=0.3，用想象回报作为标注）
- **EMA 正则化**：Critic 向自身参数的指数移动平均靠近（EMA decay=0.98），类似 target network 但用当前网络计算回报。

**Actor（策略）**：

- 使用 REINFORCE 估计器（同时支持离散和连续动作）。
- 固定熵正则化系数 η=3×10⁻⁴（跨所有域）。

**回报归一化（Return Normalization）**：

```
L(θ) = -Σ_t sg((R^λ_t - v_ψ(s_t)) / max(1, S)) · log π_θ(a_t|s_t) + η·H[π_θ(·|s_t)]
S = EMA(Per(R^λ_t, 95) - Per(R^λ_t, 5), decay=0.99)
```

用第 5 到第 95 百分位数的范围 S 归一化回报（对异常值鲁棒）；仅缩小大回报，小于阈值 L=1 的范围不缩放（避免稀疏奖励下放大噪声）。

### 关键设计亮点

1. **Symlog 变换 + Symexp Twohot 损失（跨域尺度鲁棒性）**：symlog 处理输入和向量重建目标，symexp twohot 处理奖励和价值预测，共同解决了不同域（Atari 奖励 0-10000、DMControl 奖励 0-1000、Minecraft 奖励 0-12）之间的量级差异，是实现固定超参数的核心技术。

2. **KL Balancing + Free Bits（表示学习鲁棒性）**：

   - **Free Bits**：将动力学损失和表示损失截断在 1 nat 以下（`max(1, KL[...])`），防止 KL 已很小时仍施加无意义的正则化压力；结合小的表示损失权重（β_rep=0.1），解决了不同视觉复杂度域需要不同正则化强度的困境。
   - **KL Balancing**：动力学损失（L_dyn）和表示损失（L_rep）使用不同的 stop-gradient 位置，分别训练预测器和编码器向对方靠近，权重不对称（β_dyn=1, β_rep=0.1），使预测器承担更多适应压力。

3. **可预测的模型规模扩展**：DreamerV3 的性能随参数量（12M→400M）和重放比单调提升，且更大模型需要更少环境交互，这是世界模型领域首次清晰展示的规模定律（scaling law）。

**总损失函数**（β_pred=1, β_dyn=1, β_rep=0.1）：

```
L(φ) = E_{q_φ}[ Σ_t (L_pred + L_dyn + 0.1·L_rep) ]

L_pred = -ln p_φ(x_t|z_t,h_t) - ln p_φ(r_t|z_t,h_t) - ln p_φ(c_t|z_t,h_t)
L_dyn  = max(1, KL[sg(q_φ(z_t|h_t,x_t)) || p_φ(z_t|h_t)])
L_rep  = max(1, KL[q_φ(z_t|h_t,x_t) || sg(p_φ(z_t|h_t))])
```

## 实验结果

**基准配置**（均使用单一超参数集）：

| 基准 | 任务数 | 环境步数 | 模型规模 | GPU 天数 |
|------|--------|---------|---------|---------|
| Atari | 57 | 200M | 200M 参数 | 7.7 |
| ProcGen | 16 | 50M | 200M 参数 | 16.1 |
| DMLab | 30 | 100M | 200M 参数 | 2.9 |
| Minecraft | 1 | 100M | 200M 参数 | 8.9 |
| Atari100k | 26 | 400K | 200M 参数 | 0.1 |
| BSuite | 23 | — | 200M 参数 | 0.5 |
| Proprio Control | 18 | 500K | 12M 参数 | 0.3 |
| Visual Control | 20 | 1M | 12M 参数 | 0.1 |

**Atari（57 游戏，200M 步）**：

| 方法 | Gamer Median (%) | Gamer Mean (%) |
|------|-----------------|----------------|
| PPO | 180 | 892 |
| MuZero | 693 | 3054 |
| **Dreamer** | **830** | **3381** |

**ProcGen（16 游戏，50M 步）**：

| 方法 | Normalized Mean |
|------|----------------|
| PPO（原论文调优） | 41.16 |
| PPG | 64.89 |
| **Dreamer** | **66.01** |

**DMLab（30 任务，100M 步 vs IMPALA 1B 步）**：

| 方法 | 步数 | Human Mean Capped (%) |
|------|------|----------------------|
| IMPALA | 1B | 66.3 |
| IMPALA | 100M | 31.0 |
| PPO | 100M | 35.9 |
| **Dreamer** | **100M** | **71.4** |

DreamerV3 在 100M 步超越 IMPALA 1B 步的性能（数据效率提升约 10×）。

**Minecraft Diamond（100M 步）**：

| 方法 | Return |
|------|--------|
| PPO | 5.1 |
| Rainbow | 6.3 |
| IMPALA | 7.1 |
| **Dreamer** | **9.1** |

DreamerV3 是唯一发现钻石的算法；在 0.4% 的单个 episode 中获得钻石（10 个 seed 全部发现）。相比之下，VPT 需要 720 GPU·天训练 + 人类数据；DreamerV3 仅用 1 GPU 训练 9 天，无人类数据。

**Atari100k（26 游戏，400K 步）**：

| 方法 | Gamer Mean (%) | GPU 天数 |
|------|---------------|---------|
| SPR | 62 | 0.1 |
| IRIS | 105 | 3.5 |
| **Dreamer** | **125** | **0.1** |

**Visual Control（20 任务，1M 步）**：

| 方法 | Task Mean |
|------|-----------|
| DrQ-v2 | 770 |
| **Dreamer** | **861** |

**消融实验（鲁棒性技术重要性排序）**：KL objective > Return normalization > Symexp twohot 损失 > Observation symlog。去掉重建梯度导致性能大幅下降（重建是主要表征学习信号）。

**缩放实验**：模型规模 12M → 400M 参数，性能在所有测试基准上单调提升；更高重放比可预测地提升性能；在 Crafter 和 DMLab 上均验证了上述规律。

## 局限性

- **探索能力不足**：BSuite 中 Exploration 类别（Deep Sea 任务）得分为 0，而 Boot DQN 得分 0.68，说明 DreamerV3 在需要主动探索的任务上有明显短板。
- **Minecraft episode 成功率低**：单个 episode 中获得钻石的概率仅 0.4%，仍有大量改进空间。
- **部分 Atari 游戏弱于 MuZero**：如 Alien（10977 vs 56835）、Asteroids（348684 vs 374146），说明在某些需要精确规划的游戏上仍有差距。
- **计算成本较高**：ProcGen 需要 16.1 GPU 天，Minecraft 需要 8.9 GPU 天（单 Nvidia A100）。
- **依赖重建损失**：主要依赖无监督重建，在视觉复杂但任务无关细节丰富的环境中可能浪费表征容量（与 TD-MPC2 的隐式模型相比）。
- **单域训练**：每个 agent 仍针对单一域训练，未实现跨域单一模型（与 Gato 等多任务模型不同）。
- **Skiing 等任务表现差**：Skiing 得分 -29965，接近 random（-17098）而非 human（-4337），说明某些特殊任务结构仍难以处理。

## 与同范式演进关系

**Dreamer 系列演进**：

| 版本 | 核心改进 | 主要局限 |
|------|---------|---------|
| **Dreamer (2020)** | RSSM + 潜空间 actor-critic，解析梯度 | 仅连续控制，超参数需域级调整 |
| **DreamerV2 (2021)** | 离散潜变量（categorical），超越 Atari 人类水平 | 不同视觉复杂度域的 KL 权重需手动调整 |
| **DreamerV3 (本文)** | symlog/symexp、KL balancing + free bits、twohot、回报归一化 → 固定超参数跨所有域 | 探索能力弱，依赖重建 |

**DreamerV3 对 DreamerV2 的核心改进**：DreamerV2 需要针对视觉复杂度不同的环境调整表示损失权重（3D 复杂环境需强正则化；静态背景游戏需弱正则化）。DreamerV3 通过 free bits（`max(1, KL)`）+ 小 β_rep=0.1 的组合，一次性解决了这一困境。

**与其他论文的关系**：

| 论文 | 关系 |
|------|------|
| **PlaNet (2019)** | RSSM 的来源；DreamerV3 继承并扩展了 RSSM（Block GRU、RMSNorm、SiLU） |
| **MuZero (2020)** | 同样在 Atari 上竞争；MuZero 需要在线树搜索（计算昂贵），DreamerV3 无需在线规划 |
| **TD-MPC2 (2023)** | 同期工作；DreamerV3 是生成式（重建）模型，TD-MPC2 是隐式（无解码器）模型；在连续操控任务上 TD-MPC2 更强，在离散控制（Atari）上 DreamerV3 更强 |
| **VPT (2022)** | 720 GPU·天 + 人类数据在 Minecraft 获得钻石；DreamerV3 用 1 GPU·天无人类数据实现 |
| **Gato (2022)** | 大型多任务模型，依赖专家数据；DreamerV3 无需专家数据，但每个 agent 仍单域训练 |
| **IRIS (2023)** | 基于 Transformer 的世界模型，Atari100k SOTA 之一；DreamerV3 用 0.1 GPU 天达到更高 Gamer Mean（125 vs 105） |
| **C51 / Bellemare et al. (2017)** | Twohot 离散回归的来源；DreamerV3 将其推广到奖励和价值的 symexp 变换空间 |
