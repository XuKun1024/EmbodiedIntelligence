# World Model 技术方向总览

## 四大范式对比

| 维度 | Generative | Latent Dynamics | JEPA | 3D Neural |
|------|-----------|-----------------|------|-----------|
| **Enc()** | 未显式设计为语义瓶颈，服务于像素重建 | 强（RSSM） | 强（vision encoder） | 强（3D 结构本身） |
| **Pred()** | 在 pixel 空间做预测 | 在 latent space 上，action-conditioned | 在 embedding space 上预测；部分变体 action-conditioned（如 V-JEPA 2） | 通常无 / 仅做 static 渲染 |
| **Dec()** | 核心产物（生成未来帧） | 训练时有、plan 时常可丢弃 | 无（刻意） | 核心产物（渲染像素） |
| **Reward 信号** | 无（纯自监督） | 一般有 | 无 | 无 |
| **代表工作** | Sora, Veo, Genie, GameNGen | Dreamer / DreamerV3, PlaNet, TD-MPC | I-JEPA, V-JEPA, V-JEPA 2, LeWM | NeRF, 3DGS, World Labs Marble |

---

## 各范式详解

### 1. Generative（生成式世界模型）

**核心思路**：直接在像素空间建模"下一帧长什么样"，通常基于 Diffusion / Autoregressive Transformer。

- **Enc()**：内部有较强的 encoder（如 VAE + DiT），但并非独立优化的语义压缩模块，服务于像素重建目标
- **Pred()**：预测目标在像素空间，不依赖显式 action 条件（部分工作如 Genie 2 支持 action-conditioned 生成）
- **Dec()**：生成的视频帧即核心产物
- **适用场景**：视频生成、游戏世界模拟、数据增强

**代表工作**：Sora（OpenAI）、Veo（Google）、Genie / Genie 2（DeepMind）、GameNGen

---

### 2. Latent Dynamics（潜在动态模型）

**核心思路**：在压缩的 latent space 中学习世界的状态转移，显式建模 action 对状态的影响，支持基于模型的强化学习（MBRL）。

- **Enc()**：RSSM（Recurrent State Space Model），将观测压缩为紧凑的随机 latent state
- **Pred()**：在 latent space 中做 action-conditioned 的状态转移预测
- **Dec()**：训练时重建像素用于监督，规划阶段可丢弃，大幅降低 plan 开销
- **Reward 信号**：通常有，支持在 latent space 内做 rollout + 价值估计
- **适用场景**：Model-Based RL、机器人规划、样本高效学习

**代表工作**：PlaNet、Dreamer、DreamerV3、TD-MPC

---

### 3. JEPA（Joint Embedding Predictive Architecture）

**核心思路**：在 embedding space 中预测，刻意去掉 decoder，避免模型走"像素重建捷径"，迫使网络学习高层语义表示（LeCun 提出）。

- **Enc()**：强 vision encoder（如 ViT），目标是提取语义丰富的 embedding
- **Pred()**：在 embedding space 中预测被 mask 的区域或未来帧；**原版（I-JEPA / V-JEPA）不使用 action conditioning**，V-JEPA 2 及 LeWM 扩展至 action-conditioned 场景
- **Dec()**：刻意无 decoder，是与 Generative 模型的核心区别
- **适用场景**：自监督表示学习、机器人预训练（V-JEPA 2）

**代表工作**：I-JEPA、V-JEPA、V-JEPA 2（Meta）、LeWM

---

### 4. 3D Neural（三维神经表示）

**核心思路**：用神经网络隐式或显式地表示三维场景结构，以渲染为核心产物，偏向空间理解而非时序预测。

- **Enc()**：场景本身即表示（NeRF 的 MLP / 3DGS 的 Gaussian 参数）
- **Pred()**：通常无时序预测；World Labs Marble 等新工作尝试加入动态建模
- **Dec()**：可微分渲染，将 3D 表示投影为像素图像
- **适用场景**：场景重建、新视角合成、具身导航（空间感知）

**代表工作**：NeRF、3D Gaussian Splatting（3DGS）、World Labs Marble

---

## 三维视角总结

```
                  是否在 latent/embedding 空间预测？
                         ┌─────────────────────────────┐
                    否   │  Generative     3D Neural   │  是
                         │ (pixel pred)   (rendering)  │
  是否 action-conditioned?├─────────────────────────────┤
                    是   │ Latent Dynamics    JEPA*    │
                         └─────────────────────────────┘
                    * JEPA 原版无 action；V-JEPA 2 / LeWM 扩展支持
```

---
