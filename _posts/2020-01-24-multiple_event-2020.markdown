---
layout:     post
title:      "基于门控多层次注意力机制的偏置标注网络"
subtitle:   "门控注意力"
date:       2020-01-24 10:00:00
author:     "Wwt"
header-img: "img/multiEvent/bg.png"
catalog: true
tags:   
    - NLP
---

《Collective Event Detection via a Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms》这篇18年的emnlp论文主要从句子级和文档级的注意力(attention)机制来解决事件抽取领域中标签相关性的问题。通过这篇论文主要是学习其中不同级别的注意力机制和如何编码文档级别的向量，论文介绍详细，并且提供了tensorflow的源码，于是本人研读了一番并作以总结。

传统ACE的事件抽取任务将句子中的多个事件视为独立事件，利用句子信息来识别其中的事件，但是句子中的事件通常是相互依赖的，且只利用句子信息不足以解决某类事件的歧义。因此该论文提出一种分层的、门控注意力机制和偏差标记网络，融合了句子和文档信息来解决句子中的多个事件识别问题。和先前任务不同的是，该文将事件抽取(触发词探测)任务设计为序列标注任务。实验结果表明，该文提出的网络模型取得了当时较为先进的结果。

论文中列出如下一个例子，触发词"*died*"和"*fired*"同时出现在一个句子，“died”触发了"Die"事件，“fired”触发



>In Baghdad, a camerman ***died*** when an American tank ***fired*** on the *Palestine Hotel*



了“Attack”事件。之前有关深度学习的事件抽取方法忽略了不同事件间的联系，另外文中还提出句子信息不足以解决某些事件的歧义。因此文章提出融合**句子和文档信息**，通过事件之间的依赖信息，知道“America tank” 是武器，提供给模型一定的证据来判断触发词"fired"的事件类型。但很多时候，仅通过句子信息无法识别出事件，例如



>He ***left***  the company



上面例子，模型很难判断出触发词"*left*"的事件类型，到底是离开公司还是从公司离职？但是如果通过上下文线索“*He planned to go shopping before he went home, because he got off work early today*”，将会给模型很充分的证据表明上述触发词*“left”*触发了"*Transport*"(转移)事件。

通过上述内容，论文提出一种基于层次和偏差标记的网络来解决上述问题。具体地，使用多层循环神经网络来捕获整个句子间不同事件之间的依赖关系，利用偏差目标函数来增强模型中触发词标签间的影响。另外，我们提出多层动态门控机制来融合句子和文档信息。模型框架图如下所示：

![model](/img/multiEvent/model.png)

#### 嵌入层(Embedding Layer)

本文使用word2vec作为输入的输入特征。输入一文本$d=\{s_1,s_2,...,s_i,...,s_{N_s}\}$，其中$N_s$是文本的句子数量，句子$s_i=\{w_1,w_2,...,w_t,...w_{N_w}\}$,，其中$N_w$是句子的长度，$w_t$是该句中的第$t$个词。在序列标注中，Bi-LSTM已经被证明能够有效捕捉每个词的语义信息。文中将词和实体特征编码为向量，送入到Bi-LSTM中，通过前向和后向LSTM网络层，得到每个词的隐层向量$h_{si}=[\vec{h_w},\overleftarrow{h_1}]$。然后就是Attention(注意力)层：

#### 门控注意力层(Gated Attention Layer)

##### 句子级别注意力

本文使用句子级别注意力旨在捕捉句子级的线索。对于句子中的每个候选词$w_t$，它的句子级语义信息$sh_t$通过以下计算得到：


$$
sh_t=\sum_{k=1}^
{N_w}\alpha^k_sh_k
$$


其中，$\alpha_s^k$是每个词$h_k$的权重。本文通过以下来获得 $\alpha_s^k$：


$$
\alpha^k_s=\frac{exp(z_s^k)}{\sum^{N_w}_{j=1}exp(z_s^j)}
$$


其中$z_s^k$表示第$t$个词$h_t$和第$k$个词$h_k$间的相关性，通过线性attention进行编码：


$$
z_s^k=tanh(h_tW_S{sa} h^T_k+b_{sa})
$$


其中，$W_{sa}$是权重矩阵，$b_{sa}$是偏置值。通过上述内容，本文可以得到每个词$w_t$的语义信息。

##### 文档级别注意力

类似句子级的注意力，文档级注意力机制能捕捉到重要的文档级线索。文档级语义信息$dh_i$表示第$i$个句子：


$$
dh_i=\sum^{N_s}_{k=1}\alpha_d^kh_{sk}\\
\alpha^k_d=\frac{z_d^k}{\sum^{N_s}_{j=1}exp(z_d^j)}\\
z_d^k=tanh(h_{si}W_{da}h_{sk}^T+b_{da})
$$


其中，$\alpha_d^k$是每个句子$h_{sk}$的权重，$z_d^k$是第$i$个句子$h_{si}$和第$k$个句子$h_{sk}$的相关性。$W_{da}$是权重矩阵，$b_{da}$是偏置值。与句子级信息相比，第$i$个句子有一样的文档信息$dh_i$。

另外，本文还设计了门控来动态整合第$i$个句子$s_i$中第$t$个词$w_t$中的句子级信息$sh_t$和文档级信息$dh_i$,得到它的上下文表示$cr_t$：


$$
cr_t = G_T \odot sh_t +((1-G_t)\odot dh_i)
$$


其中，$G_t$是融合门旨在编码句子线索$sh_t$和文档线索$dh_i$：


$$
G=\sigma(W_g[sh_t,dh_i]+b_g)
$$


最后，每个词$w_t$的上下文信息$cr_t$和词向量$e_t$拼接成一个向量$cr_t=[e_t,cr_t]$，作为新特征表示。

在对触发词标签进行解码时，本文提出两层标注LSTM层和一个标注attention来自动捕获事件间的隐含关系。特别地，本文还设计了一个偏置目标函数$J(\theta)$来加强触发词标签的影响，定义如下：


$$
J(\theta)= max\sum^{N_{ts}}_{j=1}\sum^{N_w}_{t=1}(logp(O_t^{y_t} \mid s_j,\theta)·I(O)+\alpha logp(O_t^{y_t} \mid s_j,\theta)·（1-I(O)))
$$


其中，$N_{ts}$是训练句子的数量，$N_w$是句子$s_j$的长度，$p(O_t^{y_t} \mid s_j,\theta)$是规范标签的概率，$y_t$是标注标签，$\alpha$是偏置权重，加大$\alpha$将会给模型的触发词标签带来更大的影响。另外，$I(O)$是模型的开关函数来区分触发词和非触发词标签的损失，定义如下：


$$
I(O)=\begin{Bmatrix}
				1 ,if \quad tag ='O'\\
				0, if \quad tag \ne "O"
   \end{Bmatrix}
$$

#### 实验结果

![res](/img/multiEvent/res.png)

实验结果表明，本文提出的模型在事件抽取上性能有很大提升，优越性体现在能从生文本中自动捕获句子的语义信息，直接预测多个事件的触发词；另外，模型还能够同时从前向和后向LSTM网络中挖掘到相邻事件之间的依赖关系。

![case](/img/multiEvent/case.png)

本文还对句子级和文档级的注意力进行可视化，例子1中，句子级信息更关注具有歧义触发词"*fired"*，其它词(tank,died and Baghdad)能够提供线索预测为*"Attack"*事件，与此同时，例子2中，周边句子"*this is ... tiresome*"，给模型提供足够的置信度预测触发词"*leave*"为"*End-Position*"事件。

#### 参考

[Collective Event Detection via a Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms](https://www.aclweb.org/anthology/D18-1158/)

[层次注意力代码](https://github.com/Alivewwt/notes/blob/master/nlp_model/doc_embedding_example/doc_embedding.py)