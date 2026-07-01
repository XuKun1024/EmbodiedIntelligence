# Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)

> **作者**：Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi
> **机构**：University of Toronto, Google Brain, DeepMind
> **年份/会议**：2020 / ICLR 2020
> **arXiv**：arXiv:1912.01603

## 一句话总结

Dreamer 在 PlaNet 的 RSSM 世界模型基础上，引入完全在潜空间中运行的 actor-critic 算法，通过解析梯度穿越多步潜在动力学反向传播来学习长视野行为，在 DeepMind Control Suite 20 个视觉控制任务上以约 3× 更快的速度超越了 D4PG 的最终性能。

## 核心贡献

- **潜空间 Actor-Critic（Behavior Learning by Latent Imagination）**：提出在世界模型的紧凑潜空间中学习参数化策略（Action Model）和价值函数（Value Model），通过将解析梯度反向传播穿越神经网络动力学（stochastic backpropagation / reparameterization trick）来高效优化策略，克服了 PlaNet 的 CEM 规划效率瓶颈。
- **λ-return 价值估计解决短视问题**：提出 V_λ 价值目标，将多步奖励累加（VR）与价值函数引导（V_N^k）混合，对想象视野长度鲁棒，解决了纯奖励最大化因视野有限导致的短视问题。
- **统一超参数、强基准性能**：在 20 个 DMControl 任务上使用同一套超参数，以约 5×10⁶ 步像素观测超越 D4PG（D4PG 需要 10⁸ 步），计算时间约 3 小时/10⁶ 步（D4PG 需约 24 小时达到相近性能）。

## 方法

### Enc()

输入为 64×64×3 像素图像（POMDP 设定）。表示模型（Representation Model）使用 CNN 接 RSSM 编码：

```
p(s_t | s_{t-1}, a_{t-1}, o_t)
```

潜在状态 s_t 为 30 维对角高斯分布，由 RSSM 的随机部分参数化。编码器条件于前一状态和动作（通过 GRU 确定性路径 h_t），以及当前观测 o_t。

**RSSM 结构**（直接继承自 PlaNet）：
```
确定性路径：h_t = f(h_{t-1}, s_{t-1}, a_{t-1})    ← GRU
随机路径：  s_t ~ p(s_t | h_t)                      ← 对角高斯，30 维
后验编码：  q(s_t | h_t, o_t)                       ← CNN + 全连接
```

### Pred()

转移模型（Transition Model）在潜空间中做 action-conditioned 预测，不依赖任何观测：

```
q(s_t | s_{t-1}, a_{t-1})   →   通过 GRU 确定性路径 h_t + 随机采样 s_t
```

**想象轨迹（Imagined Rollout）**：从真实经验的模型状态 s_t 出发，按转移模型 + 策略前向展开 H 步（H=15 连续任务，H=10 离散任务），完全在潜空间中进行，无需生成任何图像：

```
s_{τ+1} ~ q(s_{τ+1} | s_τ, a_τ),   a_τ ~ π_φ(a_τ | s_τ)
```

### Dec()

**观测解码器**：反卷积网络（deconvolutional），以 (h_t, s_t) 为输入，重建 64×64×3 图像（高斯，单位协方差）。

- **训练时**：用于计算图像重建损失 J_O^t = ln q(o_t | s_t)，驱动表征学习（与 PlaNet 相同）。
- **规划/行为学习时**：**完全不使用解码器**，actor-critic 直接在潜空间 s_t 上操作，梯度穿越潜在动力学反向传播。

**三种表示学习目标对比**（论文 Section 4）：
- **重建（Reconstruction，J_REC）**：像素级 ELBO，大多数任务最优。
- **对比估计（Contrastive/NCE，J_NCE）**：约一半任务可解，信息量理论上有上限。
- **纯奖励预测（Reward only）**：实验中不足以解决任务。

### Reward / Value 建模

**奖励模型**：全连接网络，以 s_t 为输入，输出标量奖励预测：

```
q(r_t | s_t)
```

**Value Model（价值函数 / Critic）**：
- 3 层全连接网络，每层 300 单元，ELU 激活
- 输入：潜在状态 s_τ（想象轨迹中的每一步）
- 输出：标量价值估计 v_ψ(s_τ)

**λ-return 价值目标（V_λ）**：

```
V_N^k(s_τ) = E_q [ Σ_{n=τ}^{h-1} γ^{n-τ} r_n + γ^{h-τ} v_ψ(s_h) ]   其中 h = min(τ+k, t+H)

V_λ(s_τ) = (1-λ) Σ_{n=1}^{H-1} λ^{n-1} V_N^n(s_τ) + λ^{H-1} V_N^H(s_τ)
```

参数：γ=0.99，λ=0.95。

**Action Model（策略 / Actor）**：
- 3 层全连接网络，每层 300 单元，ELU 激活
- 连续动作：输出 tanh 变换的高斯分布（均值 ×5 缩放，softplus 标准差），用重参数化采样
- 离散动作：straight-through 梯度

**训练目标**：
```
Actor：   max_φ E [ Σ_{τ=t}^{t+H} V_λ(s_τ) ]           ← 最大化价值估计
Critic：  min_ψ E [ Σ_{τ=t}^{t+H} ½(v_ψ(s_τ) - V_λ(s_τ))² ]   ← 回归价值目标（stop-gradient）
```

梯度从价值估计解析地穿越神经网络动力学反向传播（stochastic backpropagation）。

### 关键设计亮点

1. **解析梯度穿越动力学**：不同于 PlaNet 的 CEM（derivative-free 采样优化）或 REINFORCE（高方差策略梯度），Dreamer 直接通过重参数化技巧将梯度从 V_λ 反向传播穿越 H 步潜在动力学，实现高效策略优化。

2. **V_λ 对视野长度的鲁棒性**：消融实验（图4）表明，含价值模型的 Dreamer 在视野 10~45 步范围内性能稳定；纯奖励最大化（no value）对视野极敏感；PlaNet 的 CEM 规划性能随视野波动大。价值模型是解决长视野信用分配的关键。

3. **早终止建模（Early Termination）**：对有终止条件的任务，世界模型额外预测折扣因子 γ_t（二元分类器，软标签 0 和 γ），想象轨迹按累积折扣因子加权，正确处理 episode 边界。

## 实验结果

**基准**：DeepMind Control Suite 20 个连续控制任务（纯像素观测，64×64×3）。

**主要对比（5×10⁶ 步，均值）**：

| 方法 | 观测 | 环境步数 | 平均分 |
|------|------|---------|--------|
| A3C | 本体感知 | 10⁸ | 243.70 |
| D4PG | 像素 | 10⁸ | 786.32 |
| PlaNet（重跑，R=2） | 像素 | 5×10⁶ | 332.97 |
| **Dreamer** | **像素** | **5×10⁶** | **823.39** |

**部分任务详细分数（Dreamer vs D4PG）**：

| 任务 | D4PG | Dreamer |
|------|------|---------|
| Acrobot Swingup | 91.70 | **365.26** |
| Cartpole Swingup Sparse | 482.00 | **812.22** |
| Cheetah Run | 523.80 | **894.56** |
| Hopper Hop | 242.00 | **368.97** |
| Quadruped Run | — | **888.39** |
| Reacher Hard | 957.10 | **817.05** |
| Walker Run | 567.20 | **824.67** |

Dreamer 在 20 个任务中以 **20 倍更少的环境步数**平均分（823）超越 D4PG（786）。

**计算时间**：Dreamer 约 3 小时/10⁶ 步；PlaNet 约 11 小时；D4PG 达到相近性能需约 24 小时（单 Nvidia V100 GPU）。

**行为学习对比**（图7）：Dreamer 在长视野任务（Acrobot Swingup、Hopper Hop/Stand）上显著优于"No value"（纯奖励最大化）和 PlaNet；在反应性任务（Walker 系列）上三者相差不大。Dreamer 在 20 个任务中 16 个优于替代方案，4 个持平。

**表示学习对比**（图8）：重建（J_REC）> 对比估计（J_NCE）> 纯奖励预测；重建目标在大多数任务上最优。

**离散控制（Atari + DMLab，附录 C）**：在 14 个 Atari 游戏和 2 个 DMLab 关卡上测试（最多 2×10⁷ 步），部分任务表现良好，但整体尚未达到 DQN/Rainbow 水平。

## 局限性

- **表示学习是瓶颈**：论文结论明确指出"future research on representation learning can likely scale latent imagination to environments of higher visual complexity"，当前像素重建在视觉复杂度高的环境（Atari、DMLab）中不足。
- **离散控制尚不竞争**：在 Atari 上未能与 DQN/Rainbow 竞争，作者承认"agents that purely learn through world models are not yet competitive in these domains"。
- **模型误差积累**：多步想象轨迹（H=15）存在复合误差（compound error），长视野预测的准确性未深入讨论。
- **纯奖励预测不够**：当奖励稀疏、数据有限时，仅靠奖励预测无法学到充分表示，需要像素重建辅助。
- **固定动作重复 R=2**：对所有任务固定，可能对某些任务非最优。

## 与同范式演进关系

| 论文 | 与 Dreamer 的关系 |
|------|-----------------|
| **PlaNet (Hafner et al., 2019)** | Dreamer 直接继承 RSSM 世界模型；将 CEM 规划（derivative-free）替换为 actor-critic + 解析梯度，速度提升约 3.7×，性能提升约 2.5× |
| **World Models (Ha & Schmidhuber, 2018)** | 同样在隐空间训练控制器，但用 CMA-ES 进化算法；Dreamer 用解析梯度反向传播替代 |
| **SVG (Heess et al., 2015)** | 用单步模型预测降低无模型梯度方差；Dreamer 在完整潜空间多步展开并学习价值函数 |
| **DreamerV2 (Hafner et al., 2021)** | 引入离散潜变量（categorical latent，直方图分布），在 Atari 上超越人类水平，解决了 Dreamer 在离散控制上的不足 |
| **DreamerV3 (Hafner et al., 2023)** | 进一步引入 symlog 变换、KL balancing、free bits 等工程设计，实现单一超参数集跨所有任务（Atari、DMControl、Minecraft 等）通用 |
| **TD-MPC2 (2023)** | 同样在潜空间做 actor-critic，但采用 MPPI 规划 + 隐式解码器（无像素重建），在连续控制上更高效 |
