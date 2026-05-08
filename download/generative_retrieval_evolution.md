# 生成式召回（Generative Retrieval）学术演进全景

> 从 GENRE 到工业级部署：一场从"匹配"到"生成"的信息检索范式革命
>
> 涵盖 2020–2026 年关键论文、技术路线与工业界实践 | 2026-05-15

---

## 目录

- [一、什么是生成式召回](#一什么是生成式召回)
- [二、萌芽期（2020–2021）：序列生成触达检索](#二萌芽期20202021序列生成触达检索)
- [三、奠基期（2022）：模型即索引范式确立](#三奠基期2022模型即索引范式确立)
- [四、成长期（2023）：DocID 突破与推荐系统扩展](#四成长期2023docid-突破与推荐系统扩展)
- [五、繁荣期（2024）：多视角 DocID 与理论深化](#五繁荣期2024多视角-docid-与理论深化)
- [六、工业落地期（2025）：大规模部署与统一架构](#六工业落地期2025大规模部署与统一架构)
- [七、前沿探索（2025–2026）：新场景、新方法、新理论](#七前沿探索20252026新场景新方法新理论)
- [八、工业界应用全景](#八工业界应用全景)
- [九、核心挑战与未来方向](#九核心挑战与未来方向)
- [十、关键论文速查表](#十关键论文速查表)
- [十一、参考资源](#十一参考资源)

---

## 一、什么是生成式召回

### 1.1 核心思想

生成式召回（Generative Retrieval, GR）将信息检索任务重新建模为**序列生成问题**：给定用户查询（Query），模型直接自回归地生成相关文档的唯一标识符（Document Identifier, DocID），而非传统的"构建倒排索引 → 检索候选 → 打分排序"的多阶段流程。

### 1.2 传统检索 vs 生成式召回

| 维度 | 传统检索（Sparse / Dense） | 生成式召回（Generative） |
|------|---------------------------|-------------------------|
| **核心范式** | 判别式：构建索引 → 匹配 → 打分 | 生成式：端到端 Seq2Seq 生成 DocID |
| **索引方式** | 外部索引（倒排索引 / ANN 向量索引） | 模型参数即索引（Memory as Index） |
| **检索流程** | 多阶段级联：召回 → 粗排 → 精排 → 重排 | 单阶段生成，可统一召回与排序 |
| **可微分性** | 部分可微（索引构建不可微） | 全流程端到端可微 |
| **扩展性** | 索引独立扩展，支持海量语料 | 受限于模型容量和 DocID 空间 |
| **延迟** | 依赖外部索引查询（毫秒级） | 纯模型推理（需优化解码速度） |

### 1.3 关键技术要素

生成式召回的核心在于三个设计决策：

1. **DocID 设计** — 如何为每个文档分配唯一且语义丰富的标识符（数字ID、文本词元、语义ID、词集合等）
2. **模型架构** — 通常采用 Encoder-Decoder Transformer，将查询编码后自回归解码 DocID
3. **解码约束** — 通过 Constrained Beam Search 确保生成的 DocID 在语料库中有效

---

## 二、萌芽期（2020–2021）：序列生成触达检索

### 2.1 doc2query：文档扩展的生成式尝试

**论文**：doc2query: Document Expansion by Query Prediction
**作者**：Rodrigo Nogueira, Wei Wei, Jimmy Lin
**发表**：arXiv 2019 / ICML 2020

#### 要解决的问题

传统稀疏检索（如 BM25）依赖查询与文档的词汇重叠，当用户查询的表述方式与文档不同时（词汇鸿沟问题），检索效果会大幅下降。

#### 为什么可以解决

利用 Seq2Seq 模型学习"文档 → 可能的查询"的映射关系。模型见过大量查询-文档对后，能够为文档生成多种可能的查询表述，将这些生成查询附加到原文档后进行索引，相当于从多个角度"描述"了文档，从而缓解词汇鸿沟。

#### 主要网络结构

```
文档文本 → [Seq2Seq Transformer (简单版)] → 生成 N 个查询
原始文档 + 生成的查询 → [BM25 索引] → 检索
```

- 输入：文档标题 + 正文
- 输出：该文档可能被检索时使用的查询
- 模型：简单的 Encoder-Decoder Transformer
- 推理时生成 1-10 个查询，拼接到原文档后重新索引

#### 存在的问题

- 生成质量受限于模型规模（简单 Seq2Seq 表达能力有限）
- 生成查询与原文档拼接后文档变长，影响索引效率
- 本质上仍是辅助传统检索，未改变检索范式本身
- 生成的查询可能引入噪声，降低精度

---

### 2.2 docTTTTTquery：T5 增强的文档扩展

**论文**：docTTTTTquery: Document Expansion by Query Prediction with T5
**作者**：Jimmy Lin, Rodrigo Nogueira et al.
**发表**：arXiv 2020

#### 要解决的问题

doc2query 使用的简单 Seq2Seq 模型生成质量有限，生成的查询多样性不足，对 BM25 的提升有天花板。

#### 为什么可以解决

T5（Text-to-Text Transfer Transformer）是经过大规模预训练的强大 Seq2Seq 模型，具有更好的语言理解和生成能力。用 T5 替换简单 Seq2Seq 后，可以生成更自然、更多样、更贴合真实查询分布的查询。

#### 主要网络结构

```
文档文本 → [T5-base / T5-large] → 生成多个查询
原始文档 + 生成的查询 → [BM25 索引] → 检索
```

- 输入：`doc2query: {document_text}`
- 输出：可能的检索查询
- 每个文档生成最多 5 个查询
- 在 MS MARCO 上显著提升 BM25 效果

#### 存在的问题

- 仍然是辅助传统检索的"外挂"方案，未触及检索范式本身
- 生成查询的质量评估缺乏有效标准
- 查询数量与索引效率之间的 trade-off 未被系统研究
- 无法处理需要多跳推理的复杂查询

---

### 2.3 GENRE：生成式检索的开山之作

**论文**：Autoregressive Entity Retrieval (GENRE)
**作者**：Nicola De Cao, Gautier Izacard, Sebastian Riedel, Fabio Petronzi (Meta AI)
**发表**：ICLR 2021

#### 要解决的问题

传统实体检索依赖实体链接（Entity Linking）管道：先进行 mention 检测，再在知识库中候选生成，最后用判别模型排序。这种多阶段管道存在误差累积，且各阶段独立优化，无法端到端训练。

#### 为什么可以解决

GENRE 将实体检索直接建模为**自回归序列生成**：给定一个文本描述（mention context），模型直接生成目标实体的标题字符串。由于实体标题是知识库中的有效条目，生成过程天然完成了"检索"。关键创新是 **Constrained Beam Search**：在解码时约束每一步只能生成知识库中存在的实体标题前缀，确保输出一定是有效实体。

#### 主要网络结构

```
文本描述 → [BART Encoder] → 隐状态 h
h → [BART Decoder (自回归)] → 逐步生成实体标题 token
     ↓
Constrained Beam Search: 每步只保留能匹配 KB 实体标题前缀的候选
     ↓
输出: 排好序的实体列表
```

- **Encoder**：BART encoder 编码输入文本描述
- **Decoder**：BART decoder 自回归生成实体标题
- **Constrained Beam Search**：使用 Trie 数据结构存储所有 KB 实体标题，解码时每步只扩展 Trie 中存在的路径
- **训练**：给定 (mention context, 实体标题) 对，标准的 teacher-forcing 训练
- **推理**：Constrained Beam Search 生成 top-k 实体

#### 存在的问题

- 仅适用于实体检索（实体数量有限，标题较短），未推广到通用文档检索
- Constrained Beam Search 的 Trie 约束在实体规模较大时效率下降
- 模型需要记忆所有实体标题，当实体数量从万级扩展到百万级时面临挑战
- 无法处理实体标题相同但含义不同的歧义情况
- 生成式方法在训练数据稀疏的长尾实体上表现不佳

---

## 三、奠基期（2022）：模型即索引范式确立

### 3.1 DSI：可微搜索索引

**论文**：Transformer Memory as a Differentiable Search Index (DSI)
**作者**：Yi Tay, Vinh Q. Tran, Sebastian Riedel, Dara Bahri, Zhen Qin, Jiafeng Guo, Donald Metzler, William Cohen (Google)
**发表**：NeurIPS 2022

#### 要解决的问题

传统检索系统中，索引构建（如倒排索引或 ANN 向量索引）与检索模型是**分离**的——索引构建不可微，无法与检索模型联合优化。DSI 旨在消除这一分离，实现真正的端到端可微检索。

#### 为什么可以解决

DSI 的核心洞察是：**可以用 Transformer 的参数来存储整个语料库的索引信息**。具体来说，训练一个 Seq2Seq 模型，学习将查询直接映射为文档标识符（DocID）。模型参数隐式地编码了"查询 → 文档"的映射关系，相当于将索引"写入"了模型权重中。

#### 主要网络结构

```
查询 q → [T5 Encoder] → 隐状态 h
h → [T5 Decoder (自回归)] → 生成 DocID string
     ↓
Constrained Beam Search: 确保生成的 DocID 在语料库中存在
     ↓
输出: DocID → 对应文档
```

**DocID 设计（三种方案对比）**：

| 方案 | 说明 | 优缺点 |
|------|------|--------|
| **Text DocID** | 直接使用文档标题/URL 作为标识符 | 语义信息丰富，但标题可能重复或过长 |
| **Structured DocID** | 使用 `doc_{数字}` 格式 | 简洁唯一，但无语义信息，模型需死记硬背 |
| **Semantic DocID** | 用预训练模型生成文档的语义编码 | 兼顾语义和唯一性，但需要额外训练 |

**训练策略**：

- **Memorization**：直接学习 (query, docID) 对
- **Label Smoothing**：对 DocID token 应用标签平滑
- **Pre-trained Encoder + Scratch Decoder**：冻结预训练 encoder，从头训练 decoder

#### 存在的问题

- **规模瓶颈**：仅在 10K–320K 规模语料上验证，远未达到工业级（十亿级）
- **DocID 设计未优化**：三种方案各有明显缺陷，Text DocID 有重复风险，Structured DocID 无语义，Semantic DocID 需额外训练
- **解码效率低**：自回归生成 DocID 的延迟远高于传统 ANN 检索
- **增量更新困难**：新增文档需要重新训练整个模型
- **与强基线差距大**：在 MS MARCO 上远不及稠密检索（如 ColBERT）
- **Constrained Beam Search** 在大规模语料上 Trie 构建和查询开销大

---

### 3.2 NCI：神经语料索引器

**论文**：A Neural Corpus Indexer for Document Retrieval (NCI)
**作者**：Yujing Wang, Yuchen Li, Xinyu Ma, Zhiguo Wang, Yiqun Hu, Wenjie Wang, Wei Chu (ByteDance)
**发表**：NeurIPS 2022

#### 要解决的问题

DSI 的 DocID 设计存在明显缺陷：Structured DocID 无语义信息，Text DocID 可能重复。NCI 旨在设计更好的文档标识符，使生成式检索在标准基准（MS MARCO、BEIR）上具有竞争力。

#### 为什么可以解决

NCI 的核心改进在于**文档标识符的设计和训练策略**：

1. 使用更具语义信息的文本标识符（从文档内容中提取的关键短语）
2. 引入 **N-gram 约束**的 Constrained Beam Search，提高解码效率
3. 设计了更好的训练目标，使模型更好地学习查询到文档标识符的映射

#### 主要网络结构

```
查询 q → [Seq2Seq Encoder] → 隐状态 h
h → [Seq2Seq Decoder (自回归)] → 生成文档语义标识符
     ↓
N-gram Constrained Beam Search
     ↓
输出: 语义 DocID → 对应文档
```

- **DocID 设计**：从文档中提取的语义化文本片段（如关键短语），比 DSI 的 Structured DocID 更具语义信息
- **N-gram 约束解码**：使用 N-gram 索引而非完整 Trie，降低解码时的约束检查开销
- **训练**：联合优化查询编码和 DocID 生成

#### 存在的问题

- 语义化文本 DocID 仍然可能存在重复或歧义
- 在 BEIR 零样本场景下性能下降明显
- 规模扩展性未得到充分验证
- 文本 DocID 的长度不一致，影响解码效率

---

### 3.3 Ultron：模型索引器的进一步探索

**论文**：Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer
**作者**：Jung-Kai Chen, Chuan-Ju Wang, Wei Wang (Academia Sinica)
**发表**：arXiv 2022

#### 要解决的问题

DSI 和 NCI 在文档标识符编码和模型训练方面仍有改进空间。Ultron 旨在探索更优的模型索引器架构和训练方法。

#### 为什么可以解决

Ultron 提出了改进的文档标识符编码策略，将文档的语义信息更有效地压缩到标识符中，同时优化了模型的训练流程。

#### 主要网络结构

```
文档 → [语义编码器] → 压缩语义标识符
查询 → [Seq2Seq Encoder] → 隐状态
隐状态 → [Seq2Seq Decoder] → 生成语义标识符
```

- 改进的文档语义编码策略
- 优化的训练目标和课程学习策略

#### 存在的问题

- 在大规模语料上的效果未得到充分验证
- 后续扩展版本 WebUltron（IEEE TKDE 2024）才面向网页级场景
- 与同期 DSI/NCI 相比，影响力较小

---

## 四、成长期（2023）：DocID 突破与推荐系统扩展

### 4.1 TOME：两阶段文档标识符学习

**论文**：TOME: A Two-stage Approach for Model-based Retrieval
**作者**：Peitian Zhang, Zheng Liu, Yujia Zhou, Zhicheng Dou, Ji-Rong Wen (Renmin University of China)
**发表**：ACL 2023

#### 要解决的问题

此前的生成式检索方法（DSI、NCI）将 DocID 设计和检索模型训练耦合在一起，导致两个目标互相干扰：好的 DocID 需要语义丰富且唯一，好的检索模型需要高效映射查询到 DocID。联合优化时，模型可能倾向于生成"容易生成"而非"语义正确"的 DocID。

#### 为什么可以解决

TOME 的核心洞察是：**DocID 学习和检索模型训练应该解耦**。通过两阶段分离，每个阶段可以独立优化各自的目标：

- **第一阶段**：专注于学习好的 DocID（语义丰富 + 唯一区分）
- **第二阶段**：基于学到的 DocID 训练生成式检索模型

#### 主要网络结构

```
第一阶段：DocID 学习
所有文档 → [预训练 LM Encoder] → 文档嵌入
文档嵌入 → [聚类 / 量化] → 语义化 DocID (文本 token 序列)
目标: DocID 语义丰富 + 唯一区分

第二阶段：检索模型训练
查询 q → [T5 Encoder] → 隐状态 h
h → [T5 Decoder (自回归)] → 生成 DocID
目标: 给定查询，准确生成对应文档的 DocID
```

**DocID 学习的关键设计**：

1. 使用预训练语言模型（如 BERT）编码文档
2. 对文档嵌入进行聚类，同一聚类的文档获得相似的 DocID 前缀
3. 将聚类标签转换为文本 token 序列作为 DocID
4. 这种层次化的 DocID 既保留了语义信息（同语义文档有相似前缀），又保证了唯一性

#### 存在的问题

- 两阶段训练增加了工程复杂度
- 第一阶段的聚类质量直接影响第二阶段效果
- DocID 的文本表示可能较长，影响解码效率
- 聚类数量（即 DocID 空间大小）需要仔细调参

---

### 4.2 GenRet：学习语义化文档编码

**论文**：Learning to Tokenize for Generative Retrieval (GenRet)
**作者**：Zhenghao Liu, Chenyan Xiong, Xiaohui Xie, Jianxin Li, Maosong Sun, Jiawei Han (Tsinghua / Shaped.ai)
**发表**：NeurIPS 2023

#### 要解决的问题

此前的 DocID 要么使用预定义的数字编码（无语义），要么使用文档标题（可能重复），要么使用聚类标签（语义粒度粗）。GenRet 旨在学习一种**端到端的语义化文档编码**，使 DocID 本身携带丰富的文档语义信息。

#### 为什么可以解决

GenRet 的核心创新是**将文档编码（Tokenization）作为可学习的模块**，而非使用预定义的编码方案。模型通过训练自动学习如何将文档的完整语义压缩到 DocID token 序列中。

#### 主要网络结构

```
文档 d → [可学习的 Tokenizer] → 语义化 DocID tokens (t1, t2, ..., tk)
查询 q → [T5 Encoder] → 隐状态 h
h → [T5 Decoder (自回归)] → 生成 DocID tokens (t1, t2, ..., tk)
```

**可学习 Tokenizer 的设计**：

1. **词表学习**：在预训练词表的基础上，学习一组专用于 DocID 编码的 token
2. **编码目标**：DocID tokens 应最大化文档语义的保留，同时最小化不同文档间的冲突
3. **联合训练**：Tokenizer 和检索模型可以联合训练，互相促进

**训练策略**：

- 先用对比学习预训练 Tokenizer（相似文档 → 相似 DocID）
- 再用 Seq2Seq 目标训练检索模型
- 在 MS MARCO 和 6 个 BEIR 数据集上显著超越此前的生成式方法

#### 存在的问题

- 可学习 Tokenizer 的训练不稳定，需要精心设计训练策略
- DocID token 序列的长度需要在语义保留和效率之间 trade-off
- 在零样本跨域场景下性能仍有下降
- 模型参数量较大，推理开销高于传统方法

---

### 4.3 TIGER：生成式推荐的开创

**论文**：Recommender Systems with Generative Retrieval (TIGER)
**作者**：Zeyu Cui, Jianxin Ma, Changxin Tian, Xinyu Lin, Jingsen Zhang, Zhiwei Liu, Yuhao Yang, Yujia Zhou, Zhicheng Dou, Ji-Rong Wen (Kuaishou / RUC)
**发表**：NeurIPS 2023

#### 要解决的问题

传统推荐系统的召回阶段通常使用双塔模型（DSSM）或图神经网络，需要构建物品索引（如 ANN 向量索引），且召回与排序是分离的。TIGER 旨在将生成式召回引入推荐系统，用统一的生成式模型替代传统的"召回 → 排序"分离架构。

#### 为什么可以解决

TIGER 的核心思想是将推荐物品用 **Semantic ID（语义化物品标识符）** 表示，然后训练一个 Seq2Seq 模型，根据用户的历史行为序列直接生成目标物品的 Semantic ID。

#### 主要网络结构

```
物品嵌入 → [Semantic ID 学习] → 每个物品获得一个语义化 token 序列作为 ID

推理时:
用户行为序列 (i1, i2, ..., in) → [Transformer Encoder] → 隐状态 h
h → [Transformer Decoder (自回归)] → 生成目标物品的 Semantic ID
```

**Semantic ID 学习**：

1. 使用预训练的物品嵌入（如从双塔模型获得）
2. 对物品嵌入进行层次化聚类/量化
3. 每层聚类标签映射为一个 token，多层标签组成 Semantic ID
4. 例如：`[类别token] [子类别token] [细粒度token]`

**模型架构**：

- **Encoder**：Transformer encoder 编码用户行为序列
- **Decoder**：Transformer decoder 自回归生成 Semantic ID
- **Constrained Decoding**：确保生成的 Semantic ID 对应有效物品

#### 存在的问题

- Semantic ID 的层次化聚类需要预定义层数和每层的聚类数
- 物品数量远超文档数量（亿级），Semantic ID 空间设计更具挑战
- 用户行为序列的长度和稀疏性影响编码质量
- 在工业级规模（亿级物品、亿级用户）上的效果未验证
- 推理延迟在实时推荐场景中可能不满足要求

---

### 4.4 Scaling GR to Millions：大规模扩展性实证

**论文**：How Does Generative Retrieval Scale to Millions of Passages?
**作者**：Shengyao Zhuang, Linjun Shou, Jianfeng Qu, Xinyu Ma, Zhiguo Wang (ByteDance)
**发表**：EMNLP 2023

#### 要解决的问题

此前的生成式检索研究主要在 10K–320K 规模的语料上验证，远未达到工业级规模。当语料扩展到百万级时，生成式检索是否仍然可行？性能瓶颈在哪里？

#### 为什么可以解决

通过系统性实验，分析了不同 DocID 设计策略在大规模场景下的表现，为工业化应用提供了关键实验依据。

#### 主要发现

1. **DocID 设计是关键**：在大规模场景下，DocID 的语义性和区分性更加重要
2. **Structured DocID 在大规模下崩溃**：纯数字 ID 在百万级时模型几乎无法学习
3. **Text/Semantic DocID 相对稳健**：但仍存在性能下降
4. **解码效率是瓶颈**：大规模语料下 Constrained Beam Search 的延迟显著增加
5. **与稠密检索的差距随规模增大**：百万级时差距比十万级更明显

#### 存在的问题（揭示的问题）

- 生成式检索在百万级语料上仍远不及稠密检索
- 推理延迟在大规模下成为实际部署的障碍
- 模型容量需要随语料规模增长，但增长比例不明确

---

### 4.5 ROGER：排序导向的生成式检索

**论文**：ROGER: Ranking-Oriented Generative Retrieval
**作者**：Minghan Li, Chen Wu, et al.
**发表**：SIGIR 2023

#### 要解决的问题

标准生成式检索只关注"能否召回正确文档"，但实际检索系统还需要对召回结果进行排序。ROGER 旨在让生成式检索模型不仅能召回，还能直接产生排序结果。

#### 为什么可以解决

通过在 DocID 生成过程中**隐式编码排序信号**，使模型在生成多个候选 DocID 时，生成顺序本身就反映了相关性排序。

#### 主要网络结构

```
查询 q → [Encoder] → 隐状态 h
h → [Decoder] → 生成 DocID 序列 (d1, d2, d3, ...)
                    ↓
生成顺序 = 相关性排序
```

- 在训练时引入排序感知的损失函数
- 模型学习先生成最相关的文档，再生成次相关的文档
- 输出的 DocID 序列可直接作为排序结果

#### 存在的问题

- 排序质量受限于生成式模型的表达能力
- 自回归生成的顺序偏差可能影响排序公平性
- 在需要精细排序的场景下仍不如专门的排序模型

---

## 五、繁荣期（2024）：多视角 DocID 与理论深化

### 5.1 TSGen：词集合 DocID

**论文**：Generative Retrieval via Term Set Generation (TSGen)
**作者**：Peitian Zhang, Zheng Liu, Yujia Zhou, Zhicheng Dou, Ji-Rong Wen (RUC)
**发表**：SIGIR 2024

#### 要解决的问题

序列式 DocID（如 TOME、GenRet 使用的 token 序列）存在**信息瓶颈**：自回归解码时，早期生成的 token 会影响后续 token 的生成，一旦早期出错，后续全部受影响（错误累积）。此外，序列 DocID 的顺序本身可能不携带语义信息，但模型被迫学习这种顺序。

#### 为什么可以解决

TSGen 的核心创新是使用**无序词集合（Term Set）**作为 DocID，而非有序 token 序列。词集合没有顺序约束，因此可以采用**排列不变解码（Permutation-Invariant Decoding）**：在每个解码步骤，模型从剩余候选词中选择最优的一个，而非按固定顺序生成。

#### 主要网络结构

```
文档 d → [Term 提取] → 词集合 S = {t1, t2, ..., tk} (无序)

查询 q → [Encoder] → 隐状态 h
h → [Decoder] → 逐步从候选词中选择 (每步选一个，从 S 中移除)
     ↓
Permutation-Invariant Decoding:
  Step 1: 从 {t1, t2, ..., tk} 中选 t3 (最相关)
  Step 2: 从 {t1, t2, t4, ..., tk} 中选 t1
  Step 3: 从 {t2, t4, ..., tk} 中选 t5
  ...
     ↓
输出: 词集合 → 匹配文档
```

**排列不变解码的关键**：

1. 每个解码步骤，模型从所有剩余候选词中选择最优的一个
2. 选择顺序不影响最终结果（排列不变性）
3. 每个步骤模型拥有**全局视角**（看到所有剩余候选），而非仅看到前缀
4. 使用 Constrained Beam Search 确保生成的词集合在语料库中有效

#### 存在的问题

- 词集合的匹配需要额外的倒排索引或哈希查找
- 候选词数量大时，每步的选择计算量高
- 词集合的区分性可能不如序列式 DocID（信息量更少）
- 解码策略比标准自回归更复杂，工程实现难度大

---

### 5.2 SEAL：多视角文档标识符

**论文**：SEAL: Generative Retrieval with Multi-View Document Identifiers
**发表**：arXiv 2024

#### 要解决的问题

单一 DocID 只能从某个特定视角描述文档，可能遗漏文档的其他重要语义维度。例如，一篇关于"机器学习在医疗中的应用"的论文，用标题作为 DocID 可能丢失"深度学习"和"医学影像"等关键信息。

#### 为什么可以解决

SEAL 为每个文档从**不同语义视角**生成多个标识符，提供互补信息。当查询从不同角度描述文档时，至少有一个 DocID 能匹配上。

#### 主要网络结构

```
文档 d → [多视角 DocID 生成器] → {DocID_1, DocID_2, ..., DocID_m}
  视角1 (主题): DocID_1 = "machine_learning_healthcare"
  视角2 (方法): DocID_2 = "deep_learning_medical_imaging"
  视角3 (应用): DocID_3 = "clinical_diagnosis_ai"

查询 q → [Encoder-Decoder] → 生成候选 DocID
     ↓
匹配任意一个视角的 DocID 即可命中文档
```

- **多视角 DocID 生成**：使用不同的 prompt/视角引导模型从不同角度生成 DocID
- **多 DocID 索引**：每个文档对应多个 DocID，扩大了检索的命中面
- **推理**：生成式模型只需匹配其中一个 DocID 即可召回文档

#### 存在的问题

- 多 DocID 增加了索引存储和解码搜索空间
- 不同视角的 DocID 质量可能不均衡
- 多视角如何定义和选择缺乏理论指导
- 增加了训练和推理的计算开销

---

### 5.3 GR²：多级相关性生成式检索

**论文**：GR²: Generative Retrieval Meets Multi-Graded Relevance
**发表**：NeurIPS 2024

#### 要解决的问题

标准生成式检索将所有正样本文档等同看待，不区分"高度相关"和"部分相关"。但实际检索中，文档的相关性是多级的（如 TREC 的 0/1/2/3 级），模型应该能区分不同相关程度的文档。

#### 为什么可以解决

GR² 在生成式检索框架中引入**多级相关性信号**，使模型不仅知道哪些文档相关，还能区分相关程度，从而生成更精确的排序。

#### 主要网络结构

```
查询 q → [Encoder] → 隐状态 h
h → [Decoder] → 生成 DocID 序列
                ↓
训练时: 不同相关性级别的文档使用不同的损失权重
推理时: 生成顺序反映相关性级别
  高度相关文档 → 优先生成
  部分相关文档 → 后续生成
```

- **相关性感知训练**：对不同相关性级别的 DocID 应用不同的损失函数
- **标识符区分性约束**：确保不同相关性级别的文档获得不同粒度的 DocID
- **排序信号融合**：将相关性级别信息融入 DocID 生成过程

#### 存在的问题

- 多级相关性标注成本高，很多数据集只有二元标注
- 相关性级别的定义在不同场景下不一致
- 模型在细粒度相关性区分上的能力有限

---

### 5.4 GR as Multi-Vector Dense Retrieval：统一理论框架

**论文**：Generative Retrieval as Multi-Vector Dense Retrieval
**发表**：SIGIR 2024

#### 要解决的问题

生成式检索和稠密检索被视为两种截然不同的范式，缺乏统一的理论框架来理解它们之间的关系。这阻碍了对生成式检索本质的深入理解。

#### 为什么可以解决

该工作揭示了生成式检索与多向量稠密检索（如 ColBERT）之间的**深层理论联系**：两者在衡量查询-文档相关性方面共享相同的数学框架。生成式检索的自回归解码过程可以等价为多向量检索中的 token 级别交互。

#### 主要理论结论

1. 生成式检索的 DocID token 序列可以视为文档的多向量表示
2. 自回归解码的每一步对应一个 token 级别的查询-文档交互
3. Constrained Beam Search 等价于在多向量空间中的近似最近邻搜索
4. 两种范式的性能差异主要来自 DocID 设计和训练策略

#### 存在的问题

- 理论分析基于理想化假设，实际中的差异可能更大
- 未提供具体的改进方案，主要是理论贡献
- 对工业实践的指导意义有限

---

### 5.5 WebUltron：网页级生成式检索

**论文**：WebUltron: An Ultimate Retriever on Webpages Under the Model-based Indexer Paradigm
**作者**：Jung-Kai Chen, Chuan-Ju Wang, Wei Wang (Academia Sinica)
**发表**：IEEE TKDE 2024

#### 要解决的问题

此前的生成式检索研究主要在标准基准（MS MARCO、BEIR）上验证，这些基准的文档较短且领域单一。WebUltron 旨在探索生成式检索在**真实网页检索**场景中的可行性。

#### 为什么可以解决

将模型索引器范式扩展到网页级语料，针对网页文档的特点（长度不一、质量参差、HTML 噪声）进行了专门设计。

#### 主要网络结构

```
网页文档 → [HTML 清洗 + 内容提取] → 清洁文本
清洁文本 → [语义编码器] → 网页语义标识符
查询 → [Seq2Seq 模型] → 生成网页语义标识符
```

- 网页预处理 pipeline（HTML 清洗、正文提取）
- 面向长文档的语义编码策略
- 大规模网页语料上的训练和评估

#### 存在的问题

- 网页文档的异构性增加了 DocID 设计难度
- 在真实 Web 搜索场景中与工业级系统的差距仍然很大
- 网页规模（十亿级）远超实验验证规模

---

## 六、工业落地期（2025）：大规模部署与统一架构

### 6.1 OneRec：统一召回与排序

**论文**：OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment
**作者**：Yongyu Wang, et al. (Kuaishou)
**发表**：arXiv 2025

#### 要解决的问题

传统推荐系统采用"召回 → 粗排 → 精排 → 重排"的多阶段级联架构，每个阶段使用不同的模型，独立优化，存在以下问题：

1. **误差累积**：召回阶段的错误会传播到后续阶段
2. **优化目标不一致**：各阶段优化目标不同，无法全局最优
3. **工程复杂度高**：需要维护多个模型和索引系统
4. **信息损失**：每个阶段之间通过 ID 传递信息，丢失丰富的交互特征

#### 为什么可以解决

OneRec 用**单一的生成式模型**替代整个级联架构。模型直接从用户行为序列生成最终推荐结果，通过**迭代偏好对齐（Iterative Preference Alignment）**确保生成结果符合用户的真实偏好。

#### 主要网络结构

```
用户行为序列 (i1, i2, ..., in) → [Transformer Encoder] → 用户表示 h_u
h_u → [Transformer Decoder (自回归)] → 生成推荐物品 Semantic ID
     ↓
迭代偏好对齐:
  Round 1: 生成候选 → 偏好模型评估 → 反馈信号
  Round 2: 基于反馈调整生成 → 再次评估 → ...
  Round K: 收敛，输出最终推荐列表
```

**核心组件**：

1. **统一生成式模型**：一个模型完成召回和排序
2. **迭代偏好对齐**：
   - 生成候选推荐列表
   - 使用偏好模型（可以是 LLM 或专门的奖励模型）评估
   - 将评估反馈注入生成模型，调整后续生成
   - 迭代多轮直到收敛
3. **Semantic ID 设计**：层次化的物品语义标识符
4. **大规模训练**：在快手主 Feed 推荐数据上训练

**关键结果**：

- 在快手主 Feed 推荐系统中完成 A/B 测试
- **首个在真实大规模场景中显著超越现有复杂级联系统的端到端生成式模型**

#### 存在的问题

- 迭代偏好对齐增加了推理延迟（多轮生成）
- 模型训练需要大量计算资源
- 偏好模型的质量直接影响最终效果
- 架构的通用性有待在其他场景验证

---

### 6.2 Meta GEM：广告推荐基础模型

**论文**：Meta's Generative Ads Model (GEM)
**来源**：Meta Engineering Blog 2025

#### 要解决的问题

Meta 的广告推荐系统需要同时处理多种广告互动类型（点击、转化、安装等），传统方法为每种互动类型训练单独的模型，导致：

1. **模型数量爆炸**：每种互动类型一个模型，维护成本高
2. **知识不共享**：不同互动类型之间的知识无法迁移
3. **冷启动困难**：新广告互动类型需要从零训练

#### 为什么可以解决

GEM 基于 **LLM 范式**构建统一的广告推荐基础模型，通过后训练技术实现跨广告互动类型的知识迁移。

#### 主要网络结构

```
广告特征 + 用户特征 + 上下文 → [LLM-based 推荐模型] → 广告推荐结果
     ↓
后训练:
  多任务预训练 → 跨任务蒸馏 → 知识迁移
     ↓
支持多种广告互动类型: 点击、转化、安装、观看等
```

**核心技术**：

1. **LLM 规模架构**：使用 LLM 级别的模型规模和训练数据量
2. **后训练知识迁移**：
   - 在多种互动类型上联合预训练
   - 通过蒸馏过程将知识从强任务迁移到弱任务
3. **混合架构**：结合 LLM 的序列建模能力和传统推荐模型的特征处理能力

**关键结果**：

- Instagram：**+5% 广告转化率**
- Facebook：**+3% 广告转化率**
- 覆盖数十亿日活用户

#### 存在的问题

- LLM 规模模型的推理成本极高
- 蒸馏过程可能损失部分知识
- 对计算资源的需求远超传统方法
- 具体技术细节未完全公开

---

### 6.3 GLIDE：Google 搜索中的生成式召回

**论文**：GLIDE: Deploying Semantic ID-based Generative Retrieval for Large-Scale Search
**作者**：Multiple authors (Google)
**发表**：arXiv 2026

#### 要解决的问题

Google 搜索系统处理数十亿网页，传统检索管道（索引 → 召回 → 排序）虽然成熟，但存在多阶段优化不一致和工程复杂度高的问题。GLIDE 旨在将生成式召回部署到 Google 搜索系统中。

#### 为什么可以解决

GLIDE 使用 **Semantic ID** 作为文档标识符，在大规模搜索系统中部署生成式召回。通过离线指标、人工评估和 LLM 评估三重验证，确认了生成式召回的实际价值。

#### 主要网络结构

```
网页文档 → [Semantic ID 编码器] → 稀疏 Semantic ID + 细粒度物品表示
查询 → [生成式检索模型] → 生成 Semantic ID → 候选文档
     ↓
多级评估:
  1. 离线检索指标 (NDCG, Recall)
  2. 人工评估 (相关性判断)
  3. LLM 评估 (自动化质量评估)
     ↓
大规模在线 A/B 测试 → 确认业务价值
```

**核心技术**：

1. **稀疏 Semantic ID**：高效的文档标识符设计，平衡语义性和效率
2. **细粒度物品表示**：在 Semantic ID 基础上附加细粒度特征
3. **多级评估体系**：离线 + 人工 + LLM 三重验证
4. **大规模 A/B 测试**：在 Google 搜索真实流量上验证

#### 存在的问题

- 具体的 Semantic ID 设计细节未完全公开
- 与传统检索管道的集成方式不明确（是替代还是补充？）
- 推理延迟在 Google 规模下的具体数据未披露

---

### 6.4 GR4AD：快手广告系统的生成式推荐

**论文**：GR4AD: Generative Recommendation for Large-Scale Advertising
**作者**：Multiple authors (Kuaishou)
**发表**：arXiv 2026

#### 要解决的问题

广告推荐系统对实时性和吞吐量要求极高，同时需要处理复杂的广告特征（出价、预算、定向等）。将生成式推荐部署到广告系统面临独特挑战。

#### 为什么可以解决

GR4AD 在**架构、学习和推理**三个层面进行了协同设计，实现了高吞吐量的实时推理。

#### 主要网络结构

```
架构层面:
  用户特征 + 广告特征 + 上下文 → [统一生成式模型] → 广告推荐

学习层面:
  多目标优化 (点击 + 转化 + 出价)
  偏好对齐训练

推理层面:
  高效解码策略 → 满足实时性要求
  批量化推理 → 满足吞吐量要求
```

**关键设计**：

1. **架构协同**：生成式模型同时处理召回和排序
2. **学习协同**：多目标联合优化，偏好对齐
3. **推理协同**：针对广告场景优化的高效解码

**关键结果**：

- 在快手广告系统中**全面部署**
- 服务超过 **4 亿用户**
- 高吞吐量实时推理

#### 存在的问题

- 广告场景的特殊约束（出价、预算）增加了模型复杂度
- 实时推理的延迟优化细节未完全公开
- 与 OneRec 的关系和差异需要进一步分析

---

### 6.5 GenIR Survey：领域权威综述

**论文**：A Survey on Generative Information Retrieval
**作者**：Yujia Zhou, Zhicheng Dou, et al. (RUC-NLPIR)
**发表**：ACM TOIS 2025

#### 要解决的问题

生成式检索领域发展迅速，论文数量爆发式增长，缺乏系统性的梳理和总结。研究者和工程师需要一个全面的参考来理解领域全貌。

#### 为什么可以解决

该综述系统性地覆盖了生成式信息检索的所有核心议题，成为领域的权威参考。

#### 主要内容

1. **范式定义**：明确生成式检索的范畴和边界
2. **模型训练**：不同的训练策略和目标函数
3. **DocID 设计**：各种文档标识符方案的对比分析
4. **解码策略**：Constrained Beam Search 及其变体
5. **扩展性分析**：大规模场景下的性能和效率
6. **应用场景**：文档检索、推荐系统、实体检索等
7. **未来方向**：开放问题和研究机会

---

## 七、前沿探索（2025–2026）：新场景、新方法、新理论

### 7.1 GRAM：电商检索-对齐联合模型

**论文**：GRAM: Generative Retrieval and Alignment Model for E-commerce
**发表**：arXiv 2025

#### 要解决的问题

电商检索场景中，商品标识符面临独特挑战：商品数量庞大（亿级）、语义鸿沟大（用户查询"手机"可能对应"智能手机"、"移动电话"等不同表述）、商品属性结构化程度高。传统的生成式检索方法未针对电商场景优化。

#### 为什么可以解决

GRAM 使用**联合学习框架**，将生成式检索与表示对齐相结合。检索模块负责生成候选商品，对齐模块确保生成的商品与查询语义一致。

#### 主要网络结构

```
查询 q → [Encoder] → 查询表示
商品 → [Semantic ID 编码] → 商品标识符

联合训练:
  生成式检索损失: q → 生成正确的商品 DocID
  表示对齐损失: 查询表示 ≈ 相关商品表示
  对抗损失: 区分相关/不相关商品
```

#### 存在的问题

- 电商场景的商品更新频率极高，增量更新挑战更大
- 商品的多模态信息（图片、视频）未被充分利用

---

### 7.2 模型编辑实现增量文档集成

**论文**：Model Editing for New Document Integration in Generative Retrieval
**发表**：SIGIR 2025/2026

#### 要解决的问题

生成式召回将索引编码在模型参数中，新增文档需要重新训练整个模型，成本极高。这是生成式检索工业化应用的**最大障碍之一**。

#### 为什么可以解决

借鉴**模型编辑（Model Editing）**技术，通过定向修改模型参数来集成新文档，无需完全重新训练。

#### 主要网络结构

```
已有模型 M (已编码 N 个文档)
新文档 d_new → 计算需要的参数修改 Δθ
M' = M + Δθ (定向修改，不影响已有文档的检索)
```

**模型编辑策略**：

1. **定位**：确定模型中需要修改的参数（通常是与新文档 DocID 相关的权重）
2. **修改**：使用梯度或插值方法定向修改参数
3. **保持**：确保修改不影响已有文档的检索性能（局部修改）

#### 存在的问题

- 模型编辑的稳定性不保证，可能产生意外副作用
- 连续多次编辑后模型性能可能退化
- 编辑的粒度和效果需要进一步验证
- 距离工业级需求（每秒/每分钟更新）仍有差距

---

### 7.3 LIMIT：标识符歧义性分析

**论文**：Generative Retrieval Overcomes Limitations of Dense Retrieval but Struggles with Identifier Ambiguity (LIMIT)
**发表**：arXiv 2026

#### 要解决的问题

生成式检索的优势和劣势缺乏系统性的量化分析。特别是，当生成式检索失败时，失败的原因是什么？

#### 为什么可以解决

使用**合成数据集 LIMIT**，可以精确控制文档和查询的特性，从而隔离和量化不同因素对生成式检索性能的影响。

#### 主要发现

1. **优势**：生成式检索在处理语义复杂查询时优于稠密检索
2. **核心瓶颈**：**标识符歧义性（Identifier Ambiguity）**——不同文档被分配到相似 DocID，导致模型混淆
3. **规模效应**：歧义性随语料规模增大而加剧
4. **DocID 设计是关键**：好的 DocID 设计可以显著缓解歧义性

#### 存在的问题

- 合成数据集的结论在真实场景中的泛化性有待验证
- 未提出具体的歧义性缓解方案

---

### 7.4 其他前沿工作

#### 描述性+判别性 DocID（AAAI 2025）

**要解决的问题**：DocID 需要同时具备语义描述能力（让模型理解文档内容）和唯一区分能力（让模型区分不同文档）。

**方案**：提出同时优化描述性和判别性的 DocID 学习方法，通过多任务训练平衡两个目标。

#### 多级相关性 DocID（ACL 2025）

**要解决的问题**：标准 DocID 不区分文档的相关性级别。

**方案**：为不同相关性级别的文档分配不同粒度的 DocID，使模型能更精细地区分相关程度。

#### 层次化语料编码器（arXiv 2025）

**要解决的问题**：生成式检索和稠密检索各有优劣，如何结合两者？

**方案**：提出层次化编码器，上层使用生成式检索进行粗粒度召回，下层使用稠密检索进行精细排序，实现 GR+DR 混合架构。

#### 图书搜索生成式检索（arXiv 2025）

**要解决的问题**：书籍是长文档，传统的 DocID 设计不适用于书籍级别的检索。

**方案**：提出面向大纲的书籍标识符设计，利用书籍的目录结构生成层次化 DocID。

#### 跨语言生成式检索（EMNLP 2025 Findings）

**要解决的问题**：生成式检索主要在单语言场景验证，跨语言场景面临 DocID 对齐挑战。

**方案**：通过跨语言语义压缩，将不同语言的文档映射到统一的语义空间，实现跨语言生成式检索。

#### 查询规约研究（EACL 2026 Findings）

**要解决的问题**：在生成式检索框架下，查询理解面临新的挑战——模型需要将模糊查询"规约"为具体的 DocID。

**方案**：分析了生成式检索中查询规约的特殊性，提出了改进方法。

---

## 八、工业界应用全景

### 8.1 Meta — GEM 广告推荐基础模型

| 维度 | 详情 |
|------|------|
| **系统** | GEM (Generative Ads Model) |
| **场景** | 广告推荐（Instagram、Facebook） |
| **架构** | LLM 范式 + 后训练知识迁移 + 蒸馏 |
| **效果** | Instagram +5% 转化率，Facebook +3% 转化率 |
| **规模** | 数十亿日活用户 |
| **特点** | 统一多广告互动类型，跨任务知识迁移 |

### 8.2 快手 — OneRec + GR4AD

| 维度 | 详情 |
|------|------|
| **系统** | OneRec（内容推荐）+ GR4AD（广告推荐） |
| **场景** | 主 Feed 推荐 + 广告系统 |
| **架构** | 统一生成式模型 + 迭代偏好对齐 |
| **效果** | 首个在真实大规模场景超越级联系统的生成式模型 |
| **规模** | 4 亿+ 用户 |
| **特点** | 召回与排序统一，高吞吐量实时推理 |

### 8.3 Google — GLIDE 搜索

| 维度 | 详情 |
|------|------|
| **系统** | GLIDE (Generative Retrieval with Semantic IDs) |
| **场景** | Web 搜索 |
| **架构** | Semantic ID 生成式召回 + 细粒度物品表示 |
| **效果** | 通过离线+人工+LLM 三重评估验证 |
| **规模** | 数十亿网页 |
| **特点** | 多级评估体系，大规模 A/B 测试 |

### 8.4 字节跳动 — NCI + 扩展性研究

| 维度 | 详情 |
|------|------|
| **系统** | NCI (Neural Corpus Indexer) |
| **场景** | 文档检索 + 推荐系统探索 |
| **架构** | Seq2Seq + 语义化文本 DocID |
| **效果** | MS MARCO / BEIR 上超越 DSI |
| **贡献** | 奠基论文 + 百万级扩展性实证研究 |

### 8.5 腾讯 — 全模态生成式推荐竞赛

| 维度 | 详情 |
|------|------|
| **活动** | 国内首个全模态生成式推荐竞赛 |
| **参与** | 6000+ 全球学生 |
| **聚焦** | 多模态生成式大语言模型在推荐系统中的应用 |
| **意义** | 推动生成式推荐技术在学术和工业界的交流 |

### 8.6 Snap — GRID 开源框架

| 维度 | 详情 |
|------|------|
| **系统** | GRID (Generative Recommendation with Semantic IDs) |
| **场景** | 推荐系统 |
| **特点** | 开源的生成式推荐实现框架 |
| **GitHub** | snap-research/GRID |

---

## 九、核心挑战与未来方向

### 9.1 核心挑战

#### 1. DocID 设计困境

如何设计既具有**语义丰富性**又具有**唯一区分性**的文档标识符，是生成式召回的核心难题。

- 序列式 DocID 存在信息瓶颈和错误累积
- 集合式 DocID 增加了解码复杂度
- Semantic ID 需要额外的编码训练
- 多视角 DocID 增加了存储和计算开销

#### 2. 大规模扩展性

当语料规模从百万级扩展到十亿级时：

- DocID 空间急剧膨胀，模型需要记忆更多信息
- 解码搜索空间增大，推理延迟成为关键瓶颈
- 与稠密检索的性能差距可能进一步扩大

#### 3. 增量更新

传统索引可以高效地增删文档，但生成式召回将索引编码在模型参数中：

- 新增文档需要重新训练或使用模型编辑技术
- 模型编辑的稳定性和效果有待验证
- 连续多次编辑后性能可能退化

#### 4. 标识符歧义性

LIMIT 数据集的研究揭示了**标识符歧义性**是当前的主要瓶颈：

- 不同文档可能被分配到相似的 DocID
- 歧义性随语料规模增大而加剧
- 导致模型在相似文档间混淆

#### 5. 推理效率

自回归解码的串行特性导致推理速度受限：

- Constrained Beam Search 在大规模语料上延迟高
- 非自回归解码、推测解码等加速技术有待探索
- 实时推荐/搜索场景对延迟要求极高

#### 6. 多语言与跨领域

- 当前研究主要集中在英语和中文场景
- 跨语言 DocID 对齐是开放问题
- 领域自适应技术需要进一步发展

### 9.2 未来研究方向

1. **统一生成式架构**：OneRec 和 GEM 展示了统一召回与排序的潜力，未来将进一步探索端到端生成式架构，完全替代多阶段级联系统

2. **LLM 原生生成式检索**：利用大语言模型的强大序列建模和推理能力，直接在 LLM 上进行生成式检索

3. **混合检索架构**：生成式检索与稠密检索、稀疏检索的深度融合（如层次化语料编码器的 GR+DR 融合）

4. **高效增量更新机制**：模型编辑、参数高效微调（PEFT）、持续学习等技术

5. **多模态生成式检索**：扩展到图文、视频等跨模态场景

6. **推理加速与工程优化**：非自回归解码、推测解码、模型量化、KV Cache 优化

---

## 十、关键论文速查表

| 年份 | 论文 | 会议 | 核心贡献 | 解决的问题 |
|------|------|------|----------|-----------|
| 2019 | doc2query | arXiv | Seq2Seq 查询生成辅助检索 | 词汇鸿沟 |
| 2020 | docTTTTTquery | arXiv | T5 增强文档扩展 | 查询生成质量 |
| 2021 | **GENRE** | ICLR | 自回归实体检索，开山之作 | 多阶段管道误差累积 |
| 2022 | **DSI** | NeurIPS | 可微搜索索引，模型即索引 | 索引与检索分离 |
| 2022 | **NCI** | NeurIPS | 神经语料索引器，文本 DocID | DSI 的 DocID 缺陷 |
| 2022 | Ultron | arXiv | 模型索引器改进 | 编码策略优化 |
| 2023 | **TOME** | ACL | 两阶段 DocID 学习 | DocID 与检索模型耦合 |
| 2023 | **GenRet** | NeurIPS | 语义化文档编码学习 | 预定义 DocID 表达力不足 |
| 2023 | **TIGER** | NeurIPS | 生成式推荐，Semantic ID | 推荐系统召回-排序分离 |
| 2023 | Scaling GR | EMNLP | 百万级扩展性实证 | 大规模可行性未知 |
| 2023 | ROGER | SIGIR | 排序导向生成式检索 | GR 缺乏排序能力 |
| 2024 | **TSGen** | SIGIR | 词集合 DocID | 序列 DocID 信息瓶颈 |
| 2024 | SEAL | arXiv | 多视角 DocID | 单一 DocID 语义覆盖不足 |
| 2024 | **GR²** | NeurIPS | 多级相关性处理 | GR 不区分相关性级别 |
| 2024 | GR as MVDR | SIGIR | GR 与稠密检索统一理论 | 缺乏理论框架 |
| 2024 | WebUltron | IEEE TKDE | 网页级生成式检索 | 真实 Web 场景验证 |
| 2025 | **GenIR Survey** | ACM TOIS | 领域权威综述 | 缺乏系统梳理 |
| 2025 | **OneRec** | arXiv | 统一召回排序，工业验证 | 级联架构复杂度 |
| 2025 | **Meta GEM** | Meta Blog | 广告推荐基础模型 | 多互动类型模型爆炸 |
| 2025 | GRAM | arXiv | 电商检索-对齐模型 | 电商场景特殊挑战 |
| 2025 | Desc+Disc DocID | AAAI | 描述性+判别性 DocID | DocID 语义-区分性平衡 |
| 2025 | Multi-level Rel DocID | ACL | 多级相关性 DocID | 相关性粒度区分 |
| 2025 | Hierarchical Corpus | arXiv | GR+DR 混合架构 | 两范式优势互补 |
| 2025 | Book Search GR | arXiv | 图书搜索场景 | 长文档检索 |
| 2025 | Multilingual GR | EMNLP Findings | 跨语言生成式检索 | 多语言扩展 |
| 2026 | **GLIDE** | arXiv | Google 搜索大规模部署 | 工业级搜索落地 |
| 2026 | **GR4AD** | arXiv | 快手广告系统部署 | 广告场景实时推理 |
| 2026 | Model Editing GR | SIGIR | 增量文档集成 | 增量更新困难 |
| 2026 | LIMIT Analysis | arXiv | 标识符歧义性分析 | 失败原因不明 |
| 2026 | Query Spec GR | EACL Findings | 查询规约研究 | GR 查询理解挑战 |

---

## 十一、参考资源

### 综述论文

- [A Survey on Generative Information Retrieval](https://dl.acm.org/doi/10.1145/3722552) — ACM TOIS 2025（RUC-NLPIR）
- [A Survey on Generative Recommendation: Data, Model, and Tasks](https://arxiv.org/abs/2510.27157) — arXiv 2025
- [Generative Recommendation: A Survey of Models, Systems, and Applications](https://www.preprints.org/manuscript/202512.0741) — preprints.org 2025

### Tutorial

- [SIGIR 2024 Tutorial: Recent Advances in Generative IR](https://generative-ir.github.io)
- [TheWebConf 2024 Tutorial: Generative IR](https://www2024.thewebconf.org/docs/tutorial-slides/recent-advances-in-generative-ir.pdf)
- [ECIR 2024 Tutorial: Generative IR](https://ecir2024-generativeir.github.io)

### GitHub 资源

- [RUC-NLPIR/GenIR-Survey](https://github.com/RUC-NLPIR/GenIR-Survey) — 综述论文配套资源
- [awesome-generative-information-retrieval](https://github.com/gabriben/awesome-generative-information-retrieval) — 论文集合
- [awesome-generative-recommendation](https://github.com/hyp1231/awesome-generative-recommendation) — 生成式推荐论文集合
- [snap-research/GRID](https://github.com/snap-research/GRID) — Snap 生成式推荐框架

### 中文深度文章

- [生成式搜索技术深度综述：七大技术路线](https://zhuanlan.zhihu.com/p/2014300404817613189) — 知乎
- [生成式推荐工业界深度 Survey](https://www.recsys-frontier.com/article/generative-recommendation-survey) — recsys-frontier
- [Is Generative Recommendation the ChatGPT Moment of RecSys?](https://www.yuan-meng.com/posts/generative_recommendation)
- [破解集合价值建模与实时推理难题：生成式召回大模型的工业级落地](https://hub.baai.ac.cn/view/51060) — BAAI
