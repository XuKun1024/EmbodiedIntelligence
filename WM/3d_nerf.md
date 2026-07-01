# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

> **作者**：Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
> **机构**：UC Berkeley, Google Research, UC San Diego  
> **年份/会议**：2020 / ECCV 2020  
> **arXiv**：arXiv:2003.08934

## 一句话总结

NeRF 用一个 MLP 将三维坐标和视角方向映射为体积密度与颜色，通过可微分体积渲染合成任意新视角，首次实现了从稀疏 RGB 图像对复杂场景进行高保真新视角合成。

## 核心贡献

- 提出用连续 5D 神经辐射场（Neural Radiance Field）隐式表示静态三维场景，输入为空间位置 $(x,y,z)$ 和视角方向 $(\theta,\phi)$，输出为体积密度 $\sigma$ 和视角相关颜色 $c$
- 将经典体积渲染技术与神经网络相结合，构建端到端可微分渲染管线，仅以多视角 RGB 图像和相机位姿为监督
- 提出**位置编码（Positional Encoding）**，将低维坐标映射到高维傅里叶特征空间，使 MLP 能够拟合场景中的高频几何与纹理细节
- 提出**层次化体积采样（Hierarchical Volume Sampling）**策略，用粗网络（coarse network）引导细网络（fine network）在有内容的区域集中采样，提升渲染效率

## 方法

### 场景表示（Enc 等价）

场景被表示为一个全连接 MLP 网络 $F_\Theta: (x, d) \to (c, \sigma)$，其中：
- 输入为 5D 连续坐标：三维空间位置 $(x,y,z)$ + 二维视角方向 $(\theta,\phi)$（以单位向量 $d$ 表示）
- 网络结构：先用 8 层全连接层（ReLU 激活，256 通道）处理空间坐标 $x$，输出体积密度 $\sigma$ 和 256 维特征向量；再将特征向量与视角方向 $d$ 拼接后通过 1 层全连接层（ReLU，128 通道）输出 RGB 颜色 $c$
- 多视角一致性：体积密度 $\sigma$ 仅由位置 $x$ 决定，颜色 $c$ 同时依赖位置和视角，从而自然建模镜面反射等非朗伯（non-Lambertian）效果
- 整个场景的信息隐式编码在 MLP 的权重 $\Theta$ 中，存储仅需约 5 MB

### 渲染（Dec 等价）

采用经典**体积渲染（Volume Rendering）**将 3D 表示投影为 2D 图像：

1. 沿每条相机光线 $r(t) = o + td$ 在近远平面 $[t_n, t_f]$ 内采样 $N$ 个三维点
2. 将采样点坐标和视角方向输入 MLP，得到各点的颜色 $c_i$ 和密度 $\sigma_i$
3. 用离散求积公式计算像素颜色：

$$\hat{C}(r) = \sum_{i=1}^{N} T_i \left(1 - \exp(-\sigma_i \delta_i)\right) c_i, \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

其中 $T_i$ 为累积透射率，$\delta_i = t_{i+1} - t_i$ 为相邻采样点间距，alpha 值 $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$，整个过程天然可微分。

### 训练目标

训练损失为粗网络和细网络渲染颜色与真实像素颜色的均方误差之和：

$$\mathcal{L} = \sum_{r \in \mathcal{R}} \left[ \left\| \hat{C}_c(r) - C(r) \right\|_2^2 + \left\| \hat{C}_f(r) - C(r) \right\|_2^2 \right]$$

- 监督信号：已知相机位姿的多视角 RGB 图像（无需深度图或 3D 标注）
- 优化器：Adam，学习率从 $5 \times 10^{-4}$ 指数衰减至 $5 \times 10^{-5}$
- 训练规模：每 batch 随机采样 4096 条光线，粗网络每条光线 64 个采样点，细网络额外 128 个采样点
- 单场景在单张 NVIDIA V100 GPU 上训练约 100–300k 次迭代（约 1–2 天）

### 关键设计亮点

1. **位置编码（Positional Encoding）**：将坐标 $p$ 映射为 $\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$，对空间坐标取 $L=10$，对视角方向取 $L=4$。消融实验表明去掉位置编码后 PSNR 从 31.01 降至 28.77，是最关键的设计之一。

2. **层次化体积采样（Hierarchical Sampling）**：粗网络先均匀采样，根据其输出的权重分布构建分段常数 PDF，再用逆变换采样在有内容区域密集采样供细网络使用，大幅减少无效采样。

3. **视角相关颜色建模**：将颜色预测分离为与位置相关（密度）和与位置+视角相关（颜色）两部分，使模型能正确重建镜面高光等视角依赖效果，消融实验显示去掉视角依赖后 PSNR 从 31.01 降至 27.66。

## 实验结果

在三个数据集上与 SRN、Neural Volumes (NV)、LLFF 等方法对比：

| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------|--------|--------|---------|
| Diffuse Synthetic 360° | NeRF（本文） | **40.15** | **0.991** | **0.023** |
| Diffuse Synthetic 360° | LLFF | 34.38 | 0.985 | 0.048 |
| Realistic Synthetic 360° | NeRF（本文） | **31.01** | **0.947** | **0.081** |
| Realistic Synthetic 360° | LLFF | 24.88 | 0.911 | 0.114 |
| Real Forward-Facing | NeRF（本文） | **26.50** | **0.811** | 0.250 |
| Real Forward-Facing | LLFF | 24.13 | 0.798 | **0.212** |

- 在合成数据集上全面超越所有 baseline
- 在真实前向场景数据集上 PSNR 和 SSIM 优于 LLFF，但 LPIPS 略逊（0.250 vs 0.212）
- 存储极为紧凑：网络权重仅约 5 MB，比 LLFF 的体素网格（>15 GB）压缩约 3000 倍
- 训练速度：单场景需 1–2 天（约 100–300k 次迭代），推理时每帧渲染较慢（需对每个像素做 MLP 查询）

## 局限性

- **仅支持静态场景**：NeRF 对单个静态场景过拟合，无法处理动态物体或场景变化
- **训练速度慢**：单场景需 1–2 天训练，无法快速适应新场景
- **推理速度慢**：渲染新视角需要对每条光线上的大量采样点进行 MLP 前向推理，实时渲染不可行
- **泛化能力差**：每个场景需要单独训练，不能跨场景泛化
- **需要精确相机位姿**：依赖 COLMAP 等工具预先估计相机内外参，对位姿估计误差敏感
- **无法显式编辑**：场景信息隐式存储在 MLP 权重中，难以进行几何编辑或场景组合

## 与 3D 世界模型的关系

NeRF 是「3D Neural 世界模型」范式的奠基性工作，其核心价值在于：

- **隐式神经场景表示**：证明了 MLP 可以作为连续三维场景的通用表示，为后续大量工作（Instant-NGP、Mip-NeRF、3D Gaussian Splatting 等）奠定基础
- **可微分渲染**：建立了「场景表示 + 可微分渲染 + 图像重建损失」的端到端训练范式，成为后续 3D 生成模型的标准框架

在具身智能和世界模型领域，NeRF 的扩展方向包括：
- **动态 NeRF**（D-NeRF、HyperNeRF 等）：引入时间维度，建模场景动态变化，向时序世界模型演进
- **可泛化 NeRF**（pixelNeRF、IBRNet 等）：通过 encoder 提取图像特征，实现跨场景泛化，接近具身智能所需的快速场景理解能力
- **World Labs Marble 等工作**：在 NeRF/3DGS 的 3D 表示基础上引入生成模型（Diffusion 等），实现从单图或少量视角重建完整 3D 世界，并支持交互式探索，是 NeRF 向「可生成、可交互 3D 世界模型」演进的代表
- **场景重建与导航**：NeRF 提供的高质量场景重建能力可直接服务于机器人导航、操控中的空间感知任务
