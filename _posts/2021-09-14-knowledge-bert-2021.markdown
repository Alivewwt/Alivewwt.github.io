---
layout:     post
title:      "知识图谱增强语言表示"
subtitle:   "Knowledge Graph"
date:       2021-09-14 10:00:00
author:     "Wwt"
header-img: "img/kg_bert/bg.png"
catalog: true
tags:   
    - NLP
---
### Introduction

#### K-BERT的由来

当前的预训练模型（比如BERT、GPT等）往往在大规模语料上进行预训练，学习丰富的语言知识，然后在下游的特定任务上进行微调。预训练文本和微调文本之间的领域区别，BERT在领域问题上表现不佳，比如电子病历分析。

对于特殊领域的文本，一般人只能理解字面上表达的，但是专家能够根据相关领域知识进行推理。像BERT这种公共模型就像普通人一样，在通用领域表现优秀，但是在垂直领域表现不佳，除非在垂直领域进行训练，但是非常耗时耗力。

目前已经构建了大量知识图谱，将知识图谱整合到语言表示中，领域知识将能提升垂直领域的任务，并且模型具有更好的可解释性。

基于这些考虑，作者提出了一种向预训练模型中引入知识的方式，即K-BERT，其引入知识的时机是在finetune阶段。在引入知识的同时，会存在一下两个问题：

- Heterogeneous Embedding Space(HES)：通俗来讲，文本的词向量表示和KG实体的表示是通过两种独立不相关的方式分别训练得到的，造成两种向量空间独立不相关；
- Knowledge Noise(KN):向原始文本中引入太多知识有可能造成歪曲原始文本的语义。

为了解决上述两个问题，K-BERT采用了一种语句树的形式向原始文本序列中注入知识，并在预训练模型的表示空间中获取向量的表示，另外还使用了soft-position和visible matrix的方式解决了KN的问题。

#### 贡献

- 提出知识集成语言表示模型，能够解决异构空间嵌入问题和知识噪音问题；
- 通过注入知识图谱，不仅提升垂直领域的任务性能，通用领域也有提升；
- [代码公开](https://github.com/autoliuweijie/K-BERT)

### 模型结构

![1](/img/kg_bert/1.png)

从上图中看出，模型包含四个模块：knowledge layer,Embedding layer,Seeing layer和Mask-Transformer Encoder。

对于输入的文本序列，K-BERT会根据序列中存在的实体，在Knowledge Graph（KG）中找到相应的fact，例如<Cook,CEO Apple>，然后在knowledge layer 中融合，并输出相应的Sentence tree。然后将其分别输入至Embedding Layer和Seeing Layer后分别得到token对应的Embedding和Visible matrix，最后将两者传入Mask-Transformer Encoder中进行计算，并获得相应的输出向量。这些输出向量接下来被应用与下游任务，比如文本分类，序列标注等。

#### Knowledge Layer：构造Sentence tree融合KG知识

![2](/img/kg_bert/2.png)

上图展示了K-BERT整体结构，从构造sentence tree到相应的Embedding和Visible Matrix的过程。我们先来看sentence tree生成这部分，其大致分为两个步骤：

1. 找出文本序列中存在的实体，然后根据这些实体在KG中找出相应的事实三元组(fact triples)。
2. 将找出的三元组注入到原始的文本序列中，生成sentence tree。

给定一串文本序列[CLS,Time,Cook,is,visiting,Beijing,now]，序列中存在两个实体：cook和 Beijing这两个实体在KG中的fact triples分别是<Cook,CEO,Apple>、<Beijing,capital, China>和<Beijing, is_a,City>，最后将这些三元组注入到原始的文本序列中生成Sentence Tree。如上图中左下所示。

这里需要注意的是，K-BERT采用BERT作为模型骨架，BERT的输入形式是一串文本序列，并不是上述的Sentence Tree的形式，所以在实际输入的时候，我们需要对sentence tree进行拉平，形成一串文本序列。这样的操作同时会带来一些问题：

1. 直接拉平sentence tree造成句子本身穿插fact triples，破坏了句子本身的语义顺序和结构，造成信息的混乱；
2. fact triples的插入造成上述KN问题，歪曲原始句子本身的语义信息。

基于这些考虑，K-BERT提出了soft-position和visible matrix两种技术解决这些问题。这在接下来两小节中进行展开讨论。

#### Embedding layer:引入soft-position保持语句本身的语序

从上图看出，K-BERT在Embedding 层沿用了BERT Embedding layer各项相加的方式，共包含了三部分数据token embedding、position embedding和segment embedding。不过为了将Sentence tree拉平转换成一个文本序列输入给模型，K-BERT采用了一种soft-position位置编码的方式。

上图红色的标记表示的就是soft-position的索引，黑色的表示是拉平之后的绝对位置索引。在Embedding层使用的是soft-position，从而保持原始句子的正常语序。

#### Seeing layer: Mask 掉不可见的序列部分

Seeing layer 将产生一个visible matrix，其将用来控制将sentence tree拉平成序列后，序列中的词和词之间是否可见，从而保证想原始文本序列引入的fact triples不会歪曲原始句子的语义，即KN问题。

还是以上图展示的案例进行讨论，原始文本序列中的Beijing存在一个triple<Beijing,captial,china>将这triple引入到原始文本序列后在进行self-attention的时候，china仅仅能够影响Beijing这个单词，而不能影响到其他单词（比如Apple）；另外CLS同样也不能越过Cook去获得Apple的信息，否则将会造成语义信息的混乱。因此在这种情况下，需要一个visible matrix的矩阵来控制sentence tree拉平之后的各个token之间是否可见，互相之间不可见的token自然不会有影响。

#### Mask-Self-Attention:使用拉平后融入KG知识的序列进行transformer计算

由于Visible Matrix的引入，经典的transformer encoder部分无法直接去计算，需要做些改变序列的之间的可见关系进行mask，这也是Mask-Transformer名称的由来。具体如下：
$$
Q^{i+1},k^{i+1},V^{i+1}=h^iW_q,h^iW_k,h^iW_v\\S^{i+1}=softmax(\frac{Q^{i+1}K^{i+1}+M}{\sqrt{d_k}})\\h^{i+1}=S^{i+1}V^{i+1}
$$
$h^i$是第$i$个mask-self-attention的隐藏状态，$w_q$,$w_k$,$w_v$是可训练的参数，M 是visible matrix，如果对$w_k$来说$w_j$是不可见的，注意力分数$M_{jk}$设为0。

### 实验

#### 数据集

中文语料库：WikiZh、WebtextZh，中文知识图谱：CN-DBpedia、HowNet和MedicalKG。

#### 任务

#### 开放域问题

![3](/img/kg_bert/3.png)

![4](/img/kg_bert/4.png)

#### 特殊域任务

![5](/img/kg_bert/5.png)

### 参考

>[KBERT: Enabling Language Representation with Knowledge Graph](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/KBERT.html)
>
>[【论文笔记】K-BERT：Enabling Language Representation with Knowledge Graph](https://blog.csdn.net/Coding_1995/article/details/106203269)

