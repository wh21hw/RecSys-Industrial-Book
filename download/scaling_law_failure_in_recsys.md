# 推荐系统中 Scaling Law 的失效与边界：从深度扩展困境到单周期现象

> **A Survey on the Failure and Boundary of Scaling Laws in Recommendation Systems: From Depth Scaling Dilemma to the One-Epoch Phenomenon**

> 涵盖 2022–2026 年关键论文、理论分析与工业实践 | 2026-05-18

---

## 目录

- [一、引言：当 Scaling Law 遇到推荐系统](#一引言当-scaling-law-遇到推荐系统)
- [二、背景：两条截然不同的扩展路径](#二背景两条截然不同的扩展路径)
  - [2.1 LLM 中的 Scaling Law](#21-llm-中的-scaling-law)
  - [2.2 DLRM 范式与推荐系统的独特性](#22-dlrm-范式与推荐系统的独特性)
- [三、深度扩展困境：为什么更深不等于更好](#三深度扩展困境为什么更深不等于更好)
  - [3.1 现象描述：1 层即最优](#31-现象描述1-层即最优)
  - [3.2 根因分析一：Embedding 表主导参数空间](#32-根因分析一embedding-表主导参数空间)
  - [3.3 根因分析二：特征交互是组合的而非序列的](#33-根因分析二特征交互是组合的而非序列的)
  - [3.4 根因分析三：深度堆叠加剧过拟合](#34-根因分析三深度堆叠加剧过拟合)
  - [3.5 根因分析四：表格数据与序列数据的本质差异](#35-根因分析四表格数据与序列数据的本质差异)
- [四、单周期现象：数据复用的陷阱](#四单周期现象数据复用的陷阱)
  - [4.1 现象描述：One-Epoch Phenomenon](#41-现象描述one-epoch-phenomenon)
  - [4.2 根因分析一：稀疏特征的记忆化倾向](#42-根因分析一稀疏特征的记忆化倾向)
  - [4.3 根因分析二：数据分布的非平稳性](#43-根因分析二数据分布的非平稳性)
  - [4.4 根因分析三：长尾分布与样本效率](#44-根因分析三长尾分布与样本效率)
  - [4.5 工业实践：增量训练与数据新鲜度](#45-工业实践增量训练与数据新鲜度)
- [五、理论框架：理解 Scaling Law 失效的本质](#五理论框架理解-scaling-law-失效的本质)
  - [5.1 记忆化 vs 泛化：推荐系统的核心张力](#51-记忆化-vs-泛化推荐系统的核心张力)
  - [5.2 参数效率与计算效率的分离](#52-参数效率与计算效率的分离)
  - [5.3 数据效率与分布偏移的耦合](#53-数据效率与分布偏移的耦合)
- [六、突破路径：从失效到重新发现 Scaling Law](#六突破路径从失效到重新发现-scaling-law)
  - [6.1 路径一：生成式预训练桥接判别式模型（GPSD）](#61-路径一生成式预训练桥接判别式模型gpsd)
  - [6.2 路径二：基于 FM 的结构化扩展（Wukong）](#62-路径二基于-fm-的结构化扩展wukong)
  - [6.3 路径三：统一架构设计（Kunlun）](#63-路径三统一架构设计kunlun)
  - [6.4 路径四：高效序列建模（Climber）](#64-路径四高效序列建模climber)
  - [6.5 路径五：序列转导范式（HSTU / Generative Recommenders）](#65-路径五序列转导范式hstu--generative-recommenders)
  - [6.6 路径六：十亿参数推荐 Transformer（Yandex）](#66-路径六十亿参数推荐-transformeryandex)
  - [6.7 路径七：对比学习预训练缓解单周期现象](#67-路径七对比学习预训练缓解单周期现象)
  - [6.8 路径八：UniMixer 统一架构](#68-路径八unimixer-统一架构)
- [七、横向对比：各突破路径的适用边界](#七横向对比各突破路径的适用边界)
- [八、开放问题与未来方向](#八开放问题与未来方向)
- [九、关键论文速查表](#九关键论文速查表)
- [十、参考文献](#十参考文献)

---

## 一、引言：当 Scaling Law 遇到推荐系统

自 2020 年 Kaplan 等人 [1] 系统性地揭示大语言模型（LLM）的 Scaling Law 以来，"更大模型 + 更多数据 = 更好性能"已成为深度学习领域的核心信条。Chinchilla [2] 进一步确立了计算最优训练策略，GPT-4、LLaMA 等模型的成功更是将这一范式推向了极致。

然而，在推荐系统领域，一幅截然不同的图景正在浮现：

- **深度无效**：在 DLRM（Deep Learning Recommendation Model）范式中，堆叠更多 Transformer 层或 MLP 层几乎无法带来收益，1 层网络往往已经足够，甚至更深反而更差 [3][4]。
- **单周期现象（One-Epoch Phenomenon）**：工业界推荐模型通常只训练 1 个 Epoch，多 Epoch 复用数据会导致严重的过拟合，这与 LLM 中多 Epoch 训练的常规做法形成鲜明对比 [5][6]。

这两个现象共同指向一个根本性问题：**推荐系统的 Scaling Law 在何处失效，又在何处可能成立？**

本综述系统性地梳理了这一问题的研究脉络，从现象描述到根因分析，从理论框架到突破路径，试图为理解推荐系统中 Scaling Law 的边界提供一个完整的认知地图。

---

## 二、背景：两条截然不同的扩展路径

### 2.1 LLM 中的 Scaling Law

LLM 的 Scaling Law [1][2] 描述了一个简洁而强大的经验规律：

$$L(N, D, C) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_0$$

其中 $N$ 为模型参数量，$D$ 为训练数据量，$C$ 为计算量，$L$ 为损失函数。关键特性包括：

| 特性 | LLM Scaling Law |
|------|-----------------|
| **参数扩展** | 增加参数量（深度/宽度）持续降低损失 |
| **数据复用** | 多 Epoch 训练是常规操作 |
| **数据效率** | 模型越大，数据利用效率越高 |
| **可预测性** | 小规模实验可预测大规模性能 |
| **泛化性** | 更大模型泛化能力更强 |

### 2.2 DLRM 范式与推荐系统的独特性

DLRM [7] 由 Meta 于 2019 年提出，已成为工业界推荐系统的标准架构范式。其核心结构为：

```
Sparse Features → Embedding Tables → Interaction Layer (Bottom MLP / Cross Network) → Top MLP → Prediction
Dense Features  → Bottom MLP ↗
```

推荐系统与 LLM 在数据特性上存在根本差异：

| 维度 | LLM | 推荐系统 |
|------|-----|---------|
| **数据类型** | 序列文本（稠密语义） | 稀疏类别特征 + 稠密数值特征 |
| **特征空间** | 有限词汇表（~100K tokens） | 超高基数类别（亿级 ID） |
| **数据分布** | 相对平稳 | 高度非平稳（概念漂移） |
| **交互模式** | 序列组合性（sequential compositionality） | 组合交互性（combinatorial interaction） |
| **标签信号** | 丰富的语义监督 | 稀疏、噪声的隐式反馈 |
| **参数分布** | 均匀分布在各层 | 99% 集中在 Embedding 表 |

这些根本差异是理解 Scaling Law 失效的关键起点。

---

## 三、深度扩展困境：为什么更深不等于更好

### 3.1 现象描述：1 层即最优

在 DLRM 范式中，一个被广泛观察到的现象是：**增加模型深度（层数）几乎无法带来收益**。

**关键观察**：

1. **DLRM 的原始设计**：Meta 的 DLRM [7] 仅使用 1 层 Bottom MLP + 1 层 Top MLP，交互层采用简单的点积操作。后续研究表明，增加 MLP 深度并不能改善性能 [3]。
2. **DCN 的深度限制**：Deep & Cross Network (DCN) [8] 及其后续版本 DCNv2 [9] 通过 Cross Network 显式建模特征交叉，但 Cross Network 的层数通常限制在 3-5 层，更多层数反而导致性能下降。
3. **DeepFM 的深度无效**：DeepFM [10] 的 FM 部分建模二阶交互，Deep 部分建模高阶交互，但实验表明 Deep 部分的贡献远不如 FM 部分，且加深 Deep 部分收益递减。
4. **工业界实践**：Google、Meta、阿里巴巴等公司的生产级 CTR 模型普遍采用"浅层交互 + 宽度扩展"的策略，而非深度堆叠 [3][4]。

> **核心矛盾**：在 LLM 中，从 12 层增加到 96 层可以带来质的飞跃；但在推荐系统中，从 1 层增加到 4 层可能毫无收益甚至适得其反。

### 3.2 根因分析一：Embedding 表主导参数空间

推荐模型中一个被广泛忽视的事实是：**Embedding 表占据了模型参数的 99% 以上** [11][12]。

```
典型 DLRM 参数分布：
┌─────────────────────────────────────────┐
│ Embedding Tables: 99.5% (数十亿参数)     │
│ Bottom MLP:      0.3%                   │
│ Interaction:     0.1%                   │
│ Top MLP:         0.1%                   │
└─────────────────────────────────────────┘
```

**影响分析**：

- **深度增加的边际贡献极低**：增加 MLP 层数仅影响不到 1% 的参数，对模型整体表达能力的影响微乎其微。
- **Embedding 表是真正的瓶颈**：模型的表达能力主要由 Embedding 表的大小决定，而非网络的深度。
- **参数-深度解耦**：在 LLM 中，增加深度意味着成比例地增加参数；但在推荐系统中，增加深度几乎不增加有效参数。

### 3.3 根因分析二：特征交互是组合的而非序列的

"From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction" [13] 一文从信息论角度揭示了 Transformer 在推荐系统中的结构性不匹配：

**核心论点**：Transformer 假设数据的组合性是**序列的**（sequential compositionality），即信息通过逐层传递逐步构建。但推荐系统中的特征交互本质上是**组合的**（combinatorial），即多个特征之间的交互是同时发生的，而非逐层累积的。

```
LLM（序列组合性）：
  Token_1 → Layer_1 → Layer_2 → ... → Layer_N → Output
  信息逐层累积，深度有意义

推荐系统（组合交互性）：
  Feature_A ⊗ Feature_B ⊗ Feature_C → Interaction → Prediction
  交互同时发生，深度无意义
```

**实验证据**：
- DCN 的 Cross Network 通过显式建模特征交叉，在浅层即可捕获高阶交互 [8][9]。
- Wukong [14] 基于堆叠 FM（Factorization Machine）设计，证明了组合交互建模比序列建模更适合推荐系统。
- 在 CTR 预测任务中，增加 Transformer 层数无法捕获更多有意义的特征交互 [13]。

### 3.4 根因分析三：深度堆叠加剧过拟合

推荐系统面临的数据稀疏性问题使得深度模型更容易过拟合：

1. **稀疏特征的维度灾难**：用户 ID、物品 ID 等稀疏特征具有极高的基数（亿级），每个特征值可能只有极少的训练样本。
2. **深度放大过拟合**：更深的网络具有更强的拟合能力，在稀疏数据上更容易记忆噪声而非学习泛化模式。
3. **正则化效果有限**：Dropout、L2 正则化等传统方法在推荐系统中的效果远不如在 LLM 中显著 [5]。

**实证证据**：

Ardalani et al. [3] 在 Meta 的大规模 CTR 模型上系统性地研究了 Scaling Law，发现：
- 模型质量随模型大小呈幂律 + 常数形式缩放：$L(N) \approx \frac{A}{N^\alpha} + L_0$
- 但当模型大到一定程度后，**过拟合问题急剧恶化**，导致测试性能反而下降
- 这一现象在判别式推荐模型中尤为严重

### 3.5 根因分析四：表格数据与序列数据的本质差异

推荐系统的输入本质上是**表格数据**（tabular data），而非序列数据。大量研究表明，深度学习在表格数据上的优势远不如在图像、文本等感知数据上显著 [15][16]：

| 特性 | 序列/感知数据 | 表格数据 |
|------|-------------|---------|
| **特征关系** | 丰富的局部结构（空间/时间） | 稀疏、异构的特征交互 |
| **信息密度** | 高（相邻元素高度相关） | 低（特征间关系稀疏） |
| **深度价值** | 高（逐层提取层次特征） | 低（交互是扁平的） |
| **最优模型** | 深度网络 | 浅层网络 / 树模型 |

> **关键洞察**：推荐系统的核心困难不在于模型的深度或数据的规模，而在于**如何有效地建模高维稀疏特征之间的组合交互** [4]。这是一个结构问题，而非规模问题。

---

## 四、单周期现象：数据复用的陷阱

### 4.1 现象描述：One-Epoch Phenomenon

**One-Epoch Phenomenon** 是指在推荐系统的工业实践中，模型通常只训练 1 个 Epoch，多 Epoch 训练会导致严重的过拟合 [5][6]。

**典型观察**：

```
训练 Epoch 数 vs 模型性能（推荐系统）：

性能
  │     *
  │   *   *
  │  *       * ← Epoch 1 最优
  │ *           *
  │*               * ← 多 Epoch 后性能急剧下降
  └─────────────────── Epoch
```

这与 LLM 的训练形成鲜明对比：

```
训练 Epoch 数 vs 模型性能（LLM）：

性能
  │                     *
  │                   *
  │                 *
  │               *
  │             *
  │           *
  │         *
  └─────────────────── Epoch
```

**工业界实践**：
- Meta 的 DLRM 训练通常只使用 1 个 Epoch [7]
- Google 的 Wide & Deep 和 DeepFM 在生产环境中也采用单 Epoch 训练
- 阿里巴巴的 DIN/DIEN 等模型同样遵循这一模式

### 4.2 根因分析一：稀疏特征的记忆化倾向

推荐系统中的稀疏类别特征（用户 ID、物品 ID、广告 ID 等）具有极强的记忆化倾向：

1. **Embedding 表即记忆库**：Embedding 表的每一行本质上是对应特征值的"记忆"。当数据被多次遍历时，模型倾向于记忆每个特征值与标签的对应关系，而非学习泛化模式。
2. **高基数放大记忆化**：用户 ID 空间可达数十亿，每个用户可能只有几十条训练样本。多 Epoch 训练使模型过度拟合这些少量样本。
3. **记忆化 vs 泛化的失衡**：LLM 的训练数据（文本语料）具有丰富的语义结构，模型可以从中学到泛化的语言规律；而推荐系统的训练数据（点击/曝光记录）缺乏这种结构，模型更容易退化为简单的查找表。

**形式化分析**：

设 Embedding 表大小为 $|E|$，训练样本数为 $N$，则每个 Embedding 向量的平均更新次数为 $N / |E|$。在推荐系统中：
- $|E|$ 可达数十亿（用户 + 物品 + 上下文特征）
- $N$ 虽然也很大（数十亿样本），但 $N / |E|$ 可能仅为个位数
- 多 Epoch 训练使每个 Embedding 向量被反复更新，但更新方向可能不一致（因为用户偏好随时间变化），导致"记忆冲突"

### 4.3 根因分析二：数据分布的非平稳性

推荐系统的数据分布具有显著的非平稳性（non-stationarity）：

1. **用户偏好漂移**：用户的兴趣随时间变化，昨天的点击行为不能完全预测今天的偏好。
2. **物品生命周期**：新物品不断上架，旧物品逐渐过时，物品的流行度随时间剧烈变化。
3. **季节性与趋势**：节假日、热点事件等外部因素导致数据分布的系统性偏移。
4. **反馈回路**：推荐系统本身的决策会影响用户行为，形成反馈回路（feedback loop）。

**对多 Epoch 训练的影响**：

```
时间线：
  Day 1 数据 → Epoch 1 → 模型学习 Day 1 的分布
  Day 2 数据 → Epoch 2 → 模型试图学习 Day 2 的分布
                          但 Day 1 的记忆已经过时！
  Day 3 数据 → Epoch 3 → 冲突加剧...
```

当数据跨越多个时间窗口时，多 Epoch 训练使模型同时记忆了过时的和当前的分布，导致在线性能下降。

### 4.4 根因分析三：长尾分布与样本效率

推荐系统的交互数据遵循严重的长尾分布：

- **头部物品**：少数热门物品占据了大部分交互（80/20 法则甚至更极端）
- **尾部物品**：大量物品只有极少量的交互样本
- **冷启动问题**：新物品/新用户几乎没有历史数据

**多 Epoch 训练在长尾分布下的失效**：

1. **头部过拟合**：热门物品的样本在多 Epoch 中被反复学习，模型过度偏向头部物品。
2. **尾部欠拟合**：尾部物品的样本太少，即使多 Epoch 也无法提供足够的梯度信号。
3. **样本效率低下**：多 Epoch 训练将有限的计算资源浪费在重复学习头部样本上，而非探索尾部物品。

### 4.5 工业实践：增量训练与数据新鲜度

为应对单周期现象，工业界发展了一系列实践策略：

| 策略 | 描述 | 代表工作 |
|------|------|---------|
| **增量训练** | 只使用最近 N 天的数据训练，每天更新模型 | Meta DLRM [7] |
| **数据新鲜度优先** | 模型的数据新鲜度比训练充分性更重要 | Google [17] |
| **在线学习** | 实时更新模型参数，适应数据分布变化 | 各大公司 |
| **模型蒸馏** | 用大模型（多 Epoch）蒸馏小模型（单 Epoch） | 阿里巴巴 |
| **特征正则化** | 对 Embedding 施加额外的正则化约束 | DCNv2 [9] |

> **核心洞察**：在推荐系统中，**数据新鲜度 > 训练充分性**。一个用最新数据训练 1 Epoch 的模型，往往优于用历史数据训练 10 Epoch 的模型。

---

## 五、理论框架：理解 Scaling Law 失效的本质

### 5.1 记忆化 vs 泛化：推荐系统的核心张力

推荐系统的 Scaling Law 失效可以归结为一个根本性的张力：**记忆化（Memorization）vs 泛化（Generalization）**。

**LLM 的情况**：
- 训练数据（文本）具有丰富的语义结构
- 模型可以从中学到泛化的语言规律（语法、语义、推理）
- 更大的模型具有更强的泛化能力
- Scaling Law 本质上描述的是**泛化能力随规模的提升**

**推荐系统的情况**：
- 训练数据（点击/曝光记录）缺乏内在结构
- 模型的主要学习方式是**记忆**（每个 Embedding 向量记忆对应特征值的统计信息）
- 更大的模型（更深/更宽）主要增强了记忆能力，而非泛化能力
- Scaling Law 失效是因为**记忆能力不等于泛化能力**

**形式化视角**：

设训练损失为 $L_{train}$，测试损失为 $L_{test}$，泛化间隙为 $\Delta = L_{test} - L_{train}$。

- 在 LLM 中：$\Delta$ 随模型规模增大而减小（更大的模型泛化更好）
- 在推荐系统中：$\Delta$ 随模型规模增大而增大（更大的模型过拟合更严重）

### 5.2 参数效率与计算效率的分离

Kunlun [18] 从计算效率的角度分析了 Scaling Law 失效的原因：

**Model FLOPs Utilization (MFU)**：实际计算量与理论峰值计算量的比值。

在推荐系统中，MFU 通常远低于 LLM：
- Embedding 查找操作是内存密集型的，无法充分利用 GPU 的计算能力
- 稀疏特征的随机访问模式导致缓存命中率低
- 特征交互的计算密度远低于矩阵乘法

**影响**：
- 增加 Transformer 层数增加了理论计算量，但由于 MFU 低，实际有效计算量增加有限
- Kunlun 提出，**提高 MFU 是实现 Scaling Law 的前提条件**
- 通过 Generalized Dot-Product Attention (GDPA)、Hierarchical Seed Pooling (HSP) 等优化，Kunlun 在工业系统中实现了可预测的幂律缩放

### 5.3 数据效率与分布偏移的耦合

Zivic et al. [19] 在序列推荐场景中研究了 Scaling Law，发现：

1. **传统 Embedding 方案阻碍 Scaling**：当使用可训练 Embedding 表示物品时，参数量与物品数量耦合，无法独立扩展模型和数据。
2. **解耦 Embedding 是关键**：使用可学习的特征提取器替代 Embedding 查找，使参数量独立于物品数量，从而可以独立扩展模型和数据。
3. **对比学习辅助泛化**：对比学习目标可以帮助模型学习更好的物品表示，减少对记忆化的依赖。

**核心结论**：推荐系统的 Scaling Law 失效不是绝对的，而是**条件性的**——当模型架构和训练策略与数据特性匹配时，Scaling Law 可以重新显现。

---

## 六、突破路径：从失效到重新发现 Scaling Law

近年来，学术界和工业界提出了多条突破路径，试图在推荐系统中重新建立 Scaling Law。这些路径可以分为两大类：**修复判别式模型**和**转向生成式范式**。

### 6.1 路径一：生成式预训练桥接判别式模型（GPSD）

**论文**：Scaling Transformers for Discriminative Recommendation via Generative Pretraining (GPSD) [20]  
**机构**：Meta  
**发表**：KDD 2025

**核心思想**：判别式推荐模型（CTR/CVR 预测）面临严重的过拟合问题，且模型越大过拟合越严重。GPSD 提出"先生成后判别"的两阶段框架：

```
Stage 1: Generative Pretraining
  用户行为序列 → Transformer → 生成式训练（无过拟合）
  → 学习通用的用户表示

Stage 2: Discriminative Fine-tuning
  冻结 Embedding 参数 → 判别式训练（CTR/CVR）
  → 利用预训练的泛化能力
```

**关键创新**：
1. **生成式训练无过拟合**：生成式目标（next-item prediction）天然具有正则化效果，不会出现判别式训练中的过拟合问题。
2. **参数冻结策略**：冻结预训练模型的稀疏参数（Embedding），仅微调稠密参数，大幅缩小泛化间隙。
3. **Scaling Law 恢复**：通过 GPSD，判别式推荐模型首次展现出随 Transformer 深度增加而性能持续提升的 Scaling Law。

**实验结果**：
- 在工业级数据集上，GPSD 显著缩小了训练-测试泛化间隙
- 模型性能随 Transformer 深度增加而持续提升（首次在判别式推荐中观察到）
- 在线 A/B 测试中取得显著提升

**局限**：
- 需要额外的生成式预训练阶段，增加了训练成本
- 参数冻结策略可能限制模型对特定任务的适应能力

### 6.2 路径二：基于 FM 的结构化扩展（Wukong）

**论文**：Wukong: Towards a Scaling Law for Large-Scale Recommendation [14]  
**机构**：Kuaishou  
**发表**：ICML 2024 (PMLR)

**核心思想**：推荐系统需要的是**组合交互建模**而非序列建模。Wukong 基于堆叠的 Factorization Machine (FM) 设计，通过"更高更宽"的 FM 层捕获任意阶的特征交互。

```
Wukong 架构：
  Sparse Features → Embedding → FM Block × N → Linear Compress Block → Prediction
                                    ↑
                              堆叠 FM 实现高阶交互
```

**关键创新**：
1. **FM Block (FMB)**：每层 FM Block 捕获特定阶数的特征交互，堆叠多层可捕获任意阶交互。
2. **Linear Compress Block (LCB)**：将高维交互压缩为低维表示，控制计算复杂度。
3. **Scaling 策略**：通过增加 FM 层数和宽度来扩展模型，而非增加 MLP 深度。

**Scaling Law 结果**：
- Wukong 在两个数量级的模型复杂度范围内（>100 GFLOP/example）保持了幂律缩放
- 这是首个在传统推荐架构中建立 Scaling Law 的工作
- 在 6 个公开数据集上持续超越 SOTA

**局限**：
- 基于 FM 的架构在处理序列信息方面能力有限
- Scaling Law 主要体现在计算量维度，参数量维度的 Scaling 不够清晰

### 6.3 路径三：统一架构设计（Kunlun）

**论文**：Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design [18]  
**机构**：Meta / OpenAI  
**发表**：arXiv 2026

**核心思想**：Scaling Law 失效的根本原因是**计算效率低下**和**资源分配不当**。Kunlun 通过系统性的架构优化来提高 MFU，从而实现可预测的幂律缩放。

**关键优化**：
1. **Generalized Dot-Product Attention (GDPA)**：替代标准 Self-Attention，提高计算密度。
2. **Hierarchical Seed Pooling (HSP)**：高效的序列摘要机制。
3. **Computation Skip (CompSkip)**：动态跳过不必要的计算。
4. **Mixture of Wukong Experts (MoWE)**：多专家混合，提高参数利用率。

**Scaling Law 结果**：
- Kunlun 在大规模工业系统中实现了可预测的幂律缩放
- 模型性能与计算投入呈稳定的幂律关系
- 已在 Meta 的生产环境中部署

**局限**：
- 架构设计复杂，工程实现成本高
- 对特定硬件有依赖

### 6.4 路径四：高效序列建模（Climber）

**论文**：Climber: Toward Efficient Scaling Laws for Large Recommendation Models [21]  
**机构**：NetEase Cloud Music  
**发表**：CIKM 2025

**核心思想**：Transformer 在推荐系统中的 Scaling 不理想，原因在于**结构不兼容**（多源数据异构性）和**推理延迟约束**（数十毫秒级）。

**关键创新**：
1. **Multi-Scale Sequence Extraction**：通过多尺度序列提取将时间复杂度降低常数倍，实现更高效的序列长度缩放。
2. **Dynamic Temperature Modulation**：动态温度调制适应多场景、多行为模式。
3. **"单用户多物品"批处理**：创新的推理加速策略，实现 5.15× 吞吐量提升。

**Scaling Law 结果**：
- Climber 展示了更理想的 Scaling 曲线
- **首个公开记录的框架**：受控的模型扩展驱动了持续的在线指标增长（12.19% 整体提升）
- 已在网易云音乐部署，服务数千万用户

**局限**：
- 主要针对音乐推荐场景，泛化性有待验证
- 推理加速策略对特定硬件有依赖

### 6.5 路径五：序列转导范式（HSTU / Generative Recommenders）

**论文**：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU) [22]  
**机构**：Meta  
**发表**：ICML 2024

**核心思想**：将推荐任务重新建模为**序列转导任务**（sequential transduction），从 DLRM 的"特征交互"范式转向"端到端序列建模"范式。

**关键创新**：
1. **Pointwise Aggregated Attention**：高效的注意力机制，支持超长用户行为序列。
2. **生成式训练**：以 next-item prediction 为训练目标，天然避免判别式训练的过拟合。
3. **统一检索与排序**：将召回和排序统一为序列生成任务。

**Scaling Law 结果**：
- 模型质量随训练计算量呈幂律缩放，跨越三个数量级（达到 GPT-3/LLaMA-2 规模）
- 1.5 万亿参数的 Generative Recommenders 在线 A/B 测试中提升 12.4%
- **首次在推荐系统中展示了与 LLM 类似的 Scaling Law**

**局限**：
- 计算成本极高（万亿参数级别）
- 生成式推理延迟较高，不适合所有场景
- 需要大规模用户行为序列数据

### 6.6 路径六：十亿参数推荐 Transformer（Yandex）

**论文**：Scaling Recommender Transformers to One Billion Parameters [23]  
**机构**：Yandex  
**发表**：arXiv 2025

**核心思想**：HSTU 报告的最大编码器配置仅约 1.76 亿参数，远小于 LLM 的规模。Yandex 提出了将推荐 Transformer 扩展到十亿参数的方案。

**关键创新**：
1. **任务分解**：将自回归学习分解为两个子任务——反馈预测（feedback prediction）和下一物品预测（next-item prediction），每个子任务独立缩放。
2. **两阶段训练**：预训练 + 微调的流水线。
3. **上下文长度缩放**：系统性地研究了上下文长度对性能的影响。

**实验结果**：
- 成功将推荐 Transformer 扩展到十亿参数
- 在大型音乐平台上部署，总收听时间提升 2.26%，用户点赞率提升 6.37%
- **该平台历史上深度学习系统最大的推荐质量提升**

**局限**：
- 主要验证于音乐推荐场景
- 十亿参数规模仍远小于 LLM（千亿-万亿级别）

### 6.7 路径七：对比学习预训练缓解单周期现象

**论文**：Taming the One-Epoch Phenomenon in Online Recommendation [5]  
**机构**：Meta  
**发表**：CIKM 2025

**核心思想**：One-Epoch Phenomenon 的根源在于 ID Embedding 的记忆化倾向。通过对比学习预训练 ID Embedding，可以在不损失泛化能力的前提下实现多 Epoch 训练。

```
Stage 1: Contrastive Pre-training（最小化模型）
  多平台聚合数据 → 对比学习 → 预训练 ID Embedding
  → 增强数据覆盖，防止过拟合

Stage 2: Downstream Fine-tuning
  预训练 Embedding → 下游推荐模型微调
  → 利用预训练的泛化能力
```

**关键创新**：
1. **最小化模型**：使用极小的模型进行对比学习预训练，降低计算成本。
2. **负样本增强**：通过负样本降低尾部条目的有效自由度。
3. **多平台数据聚合**：利用多个推荐平台的数据增强 Embedding 的泛化能力。

**实验结果**：
- 预训练阶段的多 Epoch 训练不会导致过拟合
- 下游微调时 Embedding 的泛化能力显著提升
- 在线评估中全站参与度提升 2.2%

### 6.8 路径八：UniMixer 统一架构

**论文**：UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems [24]  
**发表**：arXiv 2026

**核心思想**：设计一个统一的架构，同时支持 Attention-based 和 TokenMixer-based 的特征交互，通过灵活的模块组合实现 Scaling Law。

**关键特性**：
- 统一处理用户历史和上下文特征
- 支持多种特征交互模式的灵活组合
- 在大规模推荐系统中展示了 Scaling Law

---

## 七、横向对比：各突破路径的适用边界

| 路径 | 核心策略 | 适用场景 | Scaling 维度 | 局限 |
|------|---------|---------|-------------|------|
| **GPSD** [20] | 生成式预训练 → 判别式微调 | CTR/CVR 预测 | 深度 | 额外预训练成本 |
| **Wukong** [14] | 堆叠 FM 捕获组合交互 | CTR 预测 | 计算量 | 序列建模能力弱 |
| **Kunlun** [18] | 架构优化提高 MFU | 大规模工业系统 | 计算效率 | 工程复杂度高 |
| **Climber** [21] | 多尺度序列 + 推理加速 | 音乐/内容推荐 | 序列长度 | 场景特异性 |
| **HSTU** [22] | 序列转导 + 生成式训练 | 检索 + 排序统一 | 计算量（3 个数量级） | 计算成本极高 |
| **Yandex** [23] | 任务分解 + 两阶段训练 | 音乐推荐 | 参数量（十亿级） | 规模仍有限 |
| **对比学习** [5] | 对比预训练 ID Embedding | 在线推荐 | Epoch 数 | 仅解决 Embedding 问题 |
| **UniMixer** [24] | 统一架构设计 | 通用推荐 | 多维度 | 较新，验证不足 |

**关键观察**：

1. **判别式 → 生成式的范式转移**：最成功的 Scaling Law 实现（HSTU、GPSD）都涉及生成式训练，这暗示生成式目标可能天然更适合推荐系统的 Scaling。
2. **架构与效率的协同设计**：Kunlun 和 Climber 表明，单纯增加模型规模不够，必须同时优化计算效率。
3. **Embedding 是核心瓶颈**：几乎所有路径都需要解决 Embedding 表的记忆化问题，无论是通过预训练（GPSD、对比学习）、替代（特征提取器）还是冻结。
4. **序列建模是突破口**：将推荐从"特征交互"重新建模为"序列建模"（HSTU、Yandex、Climber）是建立 Scaling Law 的最有效路径。

---

## 八、开放问题与未来方向

### 8.1 推荐系统的"Chinchilla 最优"是什么？

LLM 的 Chinchilla Scaling Law 确立了"20 tokens per parameter"的计算最优训练策略。推荐系统是否也存在类似的最优比例？

- **当前认知**：推荐系统的最优比例可能远高于 LLM（因为数据分布非平稳，需要更多新鲜数据）
- **开放问题**：如何定义推荐系统中的"token"？一次曝光？一个用户会话？一个特征向量？

### 8.2 判别式推荐能否真正实现 Scaling Law？

GPSD 展示了通过生成式预训练可以在判别式推荐中恢复 Scaling Law，但这是否意味着判别式推荐本身无法 Scaling？

- **核心问题**：判别式推荐的根本局限在于其训练目标（交叉熵损失）是否适合大规模扩展？
- **可能方向**：设计更适合推荐系统的自监督/对比学习目标

### 8.3 Embedding 表的 Scaling Law

Embedding 表占据了推荐模型 99% 的参数，但其 Scaling 行为研究不足：

- Embedding 维度如何随特征基数缩放？
- Embedding 表大小与模型性能之间的最优关系是什么？
- 是否存在 Embedding 压缩与 Scaling Law 的统一框架？

### 8.4 数据新鲜度与训练充分性的最优权衡

One-Epoch Phenomenon 本质上是数据新鲜度与训练充分性之间的权衡：

- 是否存在理论上的最优 Epoch 数？
- 增量学习、在线学习能否打破这一权衡？
- 数据年龄（data age）如何纳入 Scaling Law 的分析框架？

### 8.5 跨领域 Scaling Law 的统一理论

不同领域（NLP、CV、推荐、广告）的 Scaling Law 是否存在统一的理论框架？

- 推荐系统的 Scaling Law 失效是否可以归结为某些可量化的数据特性（如稀疏度、非平稳性）？
- 是否存在一个"可扩展性指数"（scalability index）来预测某个任务是否适合 Scaling？

### 8.6 小模型的 Scaling Law

当前研究主要关注大规模模型。对于资源受限的场景（移动端、边缘计算），是否存在小模型的 Scaling Law？

- 小模型在推荐系统中的 Scaling 行为是否与大模型不同？
- 知识蒸馏能否将大模型的 Scaling 优势传递给小模型？

---

## 九、关键论文速查表

| 论文 | 年份 | 机构 | 核心贡献 | 关键发现 |
|------|------|------|---------|---------|
| [Kaplan et al.] Scaling Laws for Neural Language Models | 2020 | OpenAI | 首次系统揭示 LLM Scaling Law | 损失随参数/数据/计算呈幂律下降 |
| [Hoffmann et al.] Chinchilla | 2022 | DeepMind | 计算最优训练策略 | 20 tokens/parameter 为最优比例 |
| [Naumov et al.] DLRM | 2019 | Meta | 工业级推荐模型架构标准 | Embedding + 交互层 + MLP 范式确立 |
| [Ardalani et al.] Understanding Scaling Laws for Recommendation Models | 2022 | Meta | 首次研究推荐模型 Scaling Law | 幂律 + 常数形式，但过拟合限制扩展 |
| [Wang et al.] DCN v2 | 2021 | Google | 显式特征交叉网络 | Cross Network 层数受限，深度无效 |
| [Rendle et al.] DeepFM | 2017 | Huawei | FM + Deep 双塔架构 | FM 部分贡献远大于 Deep 部分 |
| [Zhang et al.] Wukong | 2024 | Kuaishou | FM-based Scaling Law | 首个在推荐中建立 Scaling Law（传统架构） |
| [Zhai et al.] HSTU | 2024 | Meta | 万亿参数生成式推荐 | 首次展示 LLM 级别的 Scaling Law |
| [Wang et al.] GPSD | 2025 | Meta | 生成式预训练 → 判别式微调 | 判别式推荐首次恢复深度 Scaling Law |
| [Hou et al.] Kunlun | 2026 | Meta/OpenAI | 统一架构 + 效率优化 | MFU 是 Scaling Law 的前提条件 |
| [Xu et al.] Climber | 2025 | NetEase | 高效序列建模 Scaling | 首个受控扩展驱动在线持续增长 |
| [Khrylchenko et al.] Billion-Param RecSys | 2025 | Yandex | 十亿参数推荐 Transformer | 任务分解实现有效 Scaling |
| [Hsu et al.] Taming One-Epoch | 2025 | Meta | 对比学习缓解单周期现象 | 多 Epoch 训练成为可能 |
| [Zivic et al.] Scaling Sequential RecSys | 2024 | Mercado Libre | 解耦 Embedding 实现 Scaling | 特征提取器替代 Embedding 查找 |
| [NP_123] Why RecSys ≠ Deep Learning | 2026 | Towards AI | 结构性差异分析 | 推荐核心是组合交互而非深度 |

---

## 十、参考文献

[1] Kaplan, J., et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361, 2020.

[2] Hoffmann, J., et al. "Training Compute-Optimal Large Language Models." arXiv:2203.15556 (Chinchilla), NeurIPS 2022.

[3] Ardalani, N., et al. "Understanding Scaling Laws for Recommendation Models." arXiv:2208.08489, Meta, 2022.

[4] NP_123. "Why Recommendation Systems Are Structurally Different from Deep Learning." Towards AI, 2026.

[5] Hsu, Y.P., et al. "Taming the One-Epoch Phenomenon in Online Recommendation." CIKM 2025. arXiv:2508.18700.

[6] Wang, R., et al. "Exploring Scaling Laws of CTR Model for Online Performance." RecSys 2024.

[7] Naumov, M., et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems." arXiv:1906.00091, Meta, 2019.

[8] Wang, R., et al. "Deep & Cross Network for Ad Click Predictions." ADKDD 2017.

[9] Wang, R., et al. "DCN V2: Improved Deep & Cross Network." arXiv:2008.13535, 2021.

[10] Guo, H., et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI 2017.

[11] DQRM: Deep Quantized Recommendation Models. arXiv:2410.20046, 2024.

[12] EMBark: Embedding Optimization for Training Large-scale Deep Learning Recommendation Models. ACM 2024.

[13] "From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction." ResearchGate, 2024.

[14] Zhang, B., et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation." ICML 2024 (PMLR v235). arXiv:2403.02545.

[15] Gorishniy, Y., et al. "Why Deep Learning Underperforms with Tabular Data." arXiv:2110.01889, 2021.

[16] Shwartz-Ziv, R., et al. "Tabular Data: Deep Learning is Not All You Need." Information Fusion, 2022.

[17] Google. "Best Practices for Building and Deploying Recommender Systems." NVIDIA Developer Blog.

[18] Hou, B., et al. "Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design." arXiv:2602.10016, Meta/OpenAI, 2026.

[19] Zivic, P., et al. "Scaling Sequential Recommendation Models with Transformers." arXiv:2412.07585, Mercado Libre, 2024.

[20] Wang, S., et al. "Scaling Transformers for Discriminative Recommendation via Generative Pretraining (GPSD)." KDD 2025. arXiv:2506.03699.

[21] Xu, S., et al. "Climber: Toward Efficient Scaling Laws for Large Recommendation Models." CIKM 2025. arXiv:2502.09888.

[22] Zhai, J., et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024. arXiv:2402.17152.

[23] Khrylchenko, K., et al. "Scaling Recommender Transformers to One Billion Parameters." arXiv:2507.15994, Yandex, 2025.

[24] UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems. arXiv:2604.00590, 2026.

---

> **文档信息**
> - 撰写日期：2026-05-18
> - 覆盖时间范围：2019–2026
> - 核心论文数量：25+
> - 关键机构：Meta, Google, Kuaishou, NetEase, Yandex, Mercado Libre, OpenAI
