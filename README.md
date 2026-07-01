# 具身智能技术整理

本仓库系统整理具身智能（Embodied Intelligence）领域的核心技术方向，聚焦于让机器人/智能体具备感知、理解、规划与执行能力的关键方法论。

## 方向概览

| 方向 | 全称 | 一句话概括 |
|------|------|------------|
| [VLA](./VLA/) | Vision-Language-Action Model | 将视觉感知、语言理解与动作执行统一建模 |
| [WM](./WM/) | World Model | 让智能体在内部构建对物理世界的预测性表示 |
| [WAM](./WAM/) | World-Action Model | 联合建模世界动态与动作策略 |

## 目录结构

```
EmbodiedIntelligence/
├── VLA/    # Vision-Language-Action 模型
├── WM/     # World Model 世界模型
└── WAM/    # World-Action Model
```

## 三个方向的关系

```
感知 + 语言 → 动作
      ↑
   VLA：端到端的感知-语言-动作链

物理世界建模 → 预测 → 规划
      ↑
   WM：智能体对环境的内部模型

WM（世界预测） + VLA（语言指令） → 联合策略
      ↑
   WAM：世界动态与动作的协同建模
```

---

> 持续更新中。
