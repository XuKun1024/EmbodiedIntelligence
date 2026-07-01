# Veo：Google DeepMind 视频生成世界模型

> **机构**：Google DeepMind  **版本**：Veo 1（2024.5）/ Veo 2（2024.12）/ Veo 3（2025）/ Veo 3.1（2025.10）
> **形式**：产品发布 + 技术博客（无正式论文）
> **来源**：https://deepmind.google/technologies/veo/

## 一句话总结

Veo 是 Google DeepMind 发布的旗舰视频生成模型系列，以原生音视频联合生成、4K 分辨率输出和丰富的创作控制能力为核心差异点，定位于 Generative World Model 范式中的高质量视觉内容生成，并深度集成于 Google 商业生态（YouTube、Workspace、Gemini）。

## 核心贡献（各版本迭代）

- **Veo 1（2024.5）**：首次公开发布，支持 text-to-video，分辨率达 1080p，展示了对自然语言 prompt 的高度遵循能力和基础物理规律理解（液体流动、重力、碰撞）；引入 SynthID 水印对所有生成内容进行标记
- **Veo 2（2024.12）**：显著提升视频质量和时序一致性，引入摄像机运动控制（推拉摇移）、角色跨场景一致性保持，以及对真实世界物理规律的针对性训练；在 EvalBench 等评测中超越同期竞品
- **Veo 3（2025）**：核心突破为**原生音频生成**（音效、环境音、对话），实现音视频联合生成；新增 image-to-video、首尾帧插值、外绘（Outpainting）、物体增删编辑、参考图像条件生成（Ingredients to Video）等能力；在 MovieGenBench（1003 个 prompts）和 VBench I2V（355 对）上声称超越 Sora 2 Pro
- **Veo 3.1（2025.10）**：在 Veo 3 基础上进一步提升 prompt 遵循能力和物理真实感，在 MovieGenBench T2V/I2V/T2VA 三个维度均位列第一

## 方法

> **注意**：Veo 系列均无正式技术论文，以下架构分析基于 Google DeepMind 官方博客、公开技术报告及相关论文（如 Lumiere、VideoPoet）的合理推断，凡不确定之处均明确注明"未公开"。

### Enc()

**编码方式（推断）：**

Veo 的具体 Encoder 架构**未公开**。基于 Google 同期发表的相关工作（Lumiere、VideoPoet、Imagen Video），推断使用类似 **VAE（Variational Autoencoder）** 的结构将输入帧压缩至低维 latent 空间，再在 latent 空间进行扩散过程。

**输入模态：**
- 文本 prompt（T2V 模式）：通过大型语言模型（推断为 Gemini 系列）编码为文本条件向量
- 参考图像（I2V / Ingredients to Video 模式）：通过视觉 Encoder 编码为图像条件，引导场景、角色或物体的外观
- 首尾帧（插值模式）：两帧图像分别编码，作为生成过渡视频的边界条件

**压缩比例（未公开）：** 具体 latent 维度和时空压缩比例均未公开。

---

### Pred()

**预测架构（推断）：**

具体架构**未公开**。基于 Google 同期工作和行业趋势，推断 Veo 采用 **Diffusion Transformer（DiT）** 类架构，在 latent 空间执行去噪过程。相比早期 U-Net 架构，DiT 在长视频生成和高分辨率场景下具有更好的 scaling 特性。

**条件注入方式（推断）：**
- 文本条件：通过 cross-attention 机制注入 Transformer 各层
- 图像条件：通过拼接（concatenation）或 cross-attention 注入，实现参考图像引导生成
- 摄像机运动控制：具体实现方式**未公开**，推断通过专用 embedding 或 ControlNet 类机制实现
- 音频生成（Veo 3）：音视频联合建模的具体机制**未公开**，推断为音频 Decoder 与视频 Decoder 共享 latent 表示或通过跨模态 attention 实现同步

**是否 action-conditioned：** 否。Veo 不支持显式动作条件（action conditioning），摄像机控制通过文本描述或专用控制参数实现，而非 agent action 输入。

**预测目标：** 在 latent 空间预测去噪后的视频 latent，再解码为像素空间视频帧序列。

**物理理解训练（已公开）：** Google 明确指出 Veo 针对真实世界物理规律进行了专项训练，使模型能够理解液体流动、刚体碰撞、重力效果等物理现象，但具体训练方法（如物理仿真数据、物理约束损失等）**未公开**。

---

### Dec()

**解码方式（推断）：**

具体 Decoder 架构**未公开**。推断使用与 Enc() 对应的 VAE Decoder，将 latent 表示解码为像素空间的视频帧序列。

**输出规格（已公开）：**
- 分辨率：最高 4K（Veo 3 / 3.1）；Veo 2 支持至 1080p
- 单次生成时长：最长 8 秒
- 帧率：具体帧率**未公开**

**音频解码（Veo 3，推断）：** 音效、环境音和对话通过独立的音频生成模块产生，具体是否与视频 Decoder 共享参数或串行/并行生成**未公开**。

**SynthID 水印（已公开）：** 所有输出视频在解码阶段嵌入 SynthID 不可见水印，用于 AI 内容溯源和版权保护。

---

### 关键设计亮点

1. **原生音视频联合生成（Veo 3）**：区别于 Sora 等纯视频生成模型，Veo 3 首次实现原生音频生成（音效、环境音、对话），无需后处理拼接，音视频在语义和时序上高度同步。这是 Veo 系列最核心的技术差异点。

2. **参考图像条件生成（Ingredients to Video）**：用户可提供多张参考图像（人物、物体、场景），模型将其自然融合到生成视频中，实现精细的外观控制，超越纯文本 prompt 的表达能力。

3. **真实世界物理理解的针对性训练**：Google 明确指出 Veo 对物理规律进行了专项训练，在 MovieGenBench 物理真实性维度的评分上领先同类模型，体现了对具身物理规律的建模能力。

4. **SynthID 水印全链路嵌入**：所有生成内容强制嵌入不可见水印，配合有害内容过滤和版权内容检测，形成完整的安全闭环，是目前商业视频生成模型中安全机制最完善的之一。

5. **商业生态深度集成**：Veo 通过 Gemini App、Google Flow、Google Vids、YouTube 等多个产品入口向用户开放，并支持通过 Gemini API 和 Google AI Studio 进行开发者集成，形成完整的商业化路径。

## 能力亮点

以下为 Veo 相对同类模型的独特或领先能力：

| 能力 | 描述 | 相对优势 |
|------|------|----------|
| **原生音频生成** | 音效、环境音、对话与视频同步生成 | Sora 不支持；同类模型中首个原生实现 |
| **4K 分辨率输出** | 支持最高 4K 分辨率视频生成 | 超越多数竞品的 1080p 上限 |
| **Ingredients to Video** | 多张参考图像引导生成，精确控制人物/物体外观 | 精细化外观控制能力领先 |
| **摄像机控制** | 精确控制推拉摇移等专业摄像机运动 | 支持专业级运镜控制 |
| **首尾帧插值** | 在两帧图像之间生成平滑过渡视频 | 精确控制视频起止状态 |
| **外绘（Outpainting）** | 扩展原始画面边界，适配不同屏幕比例 | 支持横竖屏格式灵活转换 |
| **物体增删编辑** | 在已有视频中自然增加或移除物体 | 细粒度视频编辑能力 |
| **视频扩展/续写** | 基于末帧延续视频叙事 | 支持长视频分段生成 |
| **角色一致性** | 跨场景保持角色外观和身份 | 长视频叙事的连贯性保障 |
| **风格迁移** | 基于风格参考图复现特定视觉美学 | 精准风格控制 |
| **物理真实感** | 针对真实世界物理规律的专项训练 | MovieGenBench 物理维度评分领先 |

## 局限性

1. **短语音音频质量不稳定**：Google 官方承认，短片段内自然且连贯的口语音频生成仍在持续优化，语音连贯性和口型同步存在不稳定性（Veo 3 / 3.1 的主要已知缺陷）。

2. **单次生成时长受限（最长 8 秒）**：单次推理最长仅支持 8 秒视频，长视频需要通过视频续写功能分段拼接，缺乏全局一致性保障。

3. **架构完全不透明**：Veo 系列无正式技术论文，架构细节、训练数据、模型规模等关键信息均未公开，无法进行独立复现或科学评估。

4. **无显式动作条件（Action Conditioning）**：Veo 不支持 agent action 输入，摄像机控制依赖文本描述或专用控制参数，无法作为具身 AI 的交互式环境模拟器使用。

5. **访问受限**：Veo 3 / 3.1 目前仅通过 Google 旗下产品（Gemini App、Google Flow、Google Vids）及 API 访问，未开源，使用受商业条款约束。

6. **评测基准可信度存疑**：Veo 声称在 MovieGenBench 和 VBench I2V 上超越 Sora 2 Pro，但评测由 Google 自行进行，缺乏独立第三方验证；且评测时间点（2025年10月）与 Sora 版本对应关系存在不确定性。

7. **幻觉与物理错误**：尽管针对物理规律进行了专项训练，Veo 仍可能在复杂物理交互场景（多物体碰撞、流体动力学）中产生不符合物理规律的幻觉内容。

## 与 Generative World Model 范式的关系

### Veo 在世界模型框架中的定位

从世界模型的 Enc()–Pred()–Dec() 三元组视角来看，Veo 是一个**纯生成式视频世界模型（Generative Video World Model）**：

- **Enc()**：将输入帧/图像压缩至 latent 空间（推断为 VAE，具体未公开）
- **Pred()**：在 latent 空间执行扩散预测，以文本/图像为条件，**无显式 action conditioning**
- **Dec()**：将预测的 latent 解码为像素空间视频帧，生成视频是核心产物

Veo 的核心目标是**高质量视觉内容生成**，而非构建可供 agent 交互的环境模拟器。

---

### 与 Sora（OpenAI）的对比

| 维度 | Veo 3.1 | Sora 2 Pro |
|------|---------|------------|
| 音频生成 | 原生支持（音效/环境音/对话） | 不支持 |
| 最高分辨率 | 4K | 1080p |
| 单次时长 | 最长 8 秒 | 最长 20 秒 |
| 摄像机控制 | 支持 | 支持 |
| 商业集成 | YouTube/Workspace/Gemini | ChatGPT/Sora.com |
| 安全机制 | SynthID 水印 + 多层过滤 | 元数据标记 |
| 技术透明度 | 无正式论文 | 技术报告（有限披露） |
| 开放程度 | API 访问 | API 访问 |

**核心差异**：Veo 3 的原生音频生成是对 Sora 的最显著差异点；Veo 更强调商业生态集成和安全机制；Sora 在单次生成时长上更有优势（20 秒 vs 8 秒）。

---

### 与 Genie（Google DeepMind）的对比

Genie 是同机构的**可交互世界模型**，与 Veo 定位截然不同：

- **Genie**：无监督学习隐式动作空间（Latent Action Model），支持帧级交互控制，目标是构建可供 agent 探索的虚拟环境；生成质量和分辨率较低（160×90），约 1 FPS
- **Veo**：无 action conditioning，目标是高质量视频内容生成；4K 分辨率，不支持实时交互

两者代表了 Generative World Model 的两个方向：Genie 追求**可控性和可交互性**，Veo 追求**视觉质量和创作表达能力**。

---

### 与 GameNGen（Google Research）的对比

GameNGen 是**实时游戏引擎模拟器**，同样基于扩散模型，但与 Veo 定位完全不同：

- **GameNGen**：显式 action conditioning（键盘输入），20 FPS 实时运行，目标是模拟特定游戏（DOOM）的交互逻辑；分辨率低（320×240），无泛化能力
- **Veo**：无 action conditioning，高分辨率（4K），强泛化能力，目标是通用视频内容生成

GameNGen 体现了 World Model 的**交互模拟**范式，Veo 体现了**视觉生成**范式。

---

### 在具身智能中的潜在价值与局限

**潜在价值：**
- 作为**视觉数据合成器**：Veo 的高质量视频生成能力可用于生成具身 AI 训练所需的多样化视觉数据
- 作为**物理理解基准**：Veo 在物理真实感上的针对性训练，提示了视频生成模型作为物理规律学习载体的潜力
- **音视频联合生成**（Veo 3）：为具身环境的多模态感知数据合成提供了新的可能性

**核心局限（具身视角）：**
- **无 action conditioning**：无法接受 agent 动作输入，不能作为交互式环境模拟器，不能用于 model-based RL 的 rollout 生成
- **无状态追踪**：不维护显式的环境状态表示，无法保证长时序的物理一致性
- **非实时**：推理速度远低于实时要求，无法支持 agent 在线交互
- **封闭系统**：无法开放权重或 API 进行定制化的具身场景微调

**总结**：Veo 在具身智能中的定位是**离线视觉数据生成工具**，而非可供 agent 交互的世界模型。要实现具身 AI 所需的交互式环境模拟，需要在 Veo 类架构基础上引入显式 action conditioning（参考 GameNGen）和状态一致性机制（参考 Genie）。

## 参考来源

- Google DeepMind 官方页面：https://deepmind.google/technologies/veo/
- Google DeepMind 模型页面：https://deepmind.google/models/veo/
- SynthID 技术介绍：https://deepmind.google/technologies/synthid/
- MovieGenBench（Meta）：https://ai.meta.com/research/movie-gen/
- VBench I2V 基准：https://huggingface.co/spaces/Vchitect/VBench_Leaderboard
- 相关架构参考论文：
  - Lumiere（Google Research, 2024）：arXiv:2401.12945
  - VideoPoet（Google Research, 2024）：arXiv:2312.14125
  - Scalable Diffusion Transformers（Peebles & Xie, 2023）：arXiv:2212.09748
