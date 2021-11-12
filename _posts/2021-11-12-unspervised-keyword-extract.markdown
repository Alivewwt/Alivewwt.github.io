---
layout:     post
title:      "基于无监督的关键词抽取"
subtitle:   "key word extraction"
date:       2021-11-12 10:00:00
author:     "Wwt"
header-img: "img/keyword-extract/bg.png"
catalog: true
tags:   
    - NLP
---



>本文参考自[关键词抽取算法](https://zhuanlan.zhihu.com/p/377737998)，部分有修改。

### 任务

关键词提取是从文本检索关键字或关键短语。这些关键词从文本中短语中选择出来并且代表了文档的主题。在本篇短文中我们将介绍几种常见的关键词抽取方法。

自动从文档中提取关键字是从文本文档中选择最常用和最重要的单词或短语的启发式方法。关键字提取是自然语言处理领域中的一个重要任务。

通过抽取关键字，有以下三个方面优点：

1. 节省时间，根据关键词，可以决定文本的主题（例如文章）是否引起用户的兴趣以及最终阅读。关键字向用户提供了该篇文章或文档主要内容摘要。
2. 查找相关文档，大量文章的出现使得我们不可能全部进行阅读。关键词提取算法可以帮助我们找到相关文章。关键字提取算法还可以自动构建书籍、出版物或索引。
3. 提取关键词可以支持机器学习，找到描述文本的最相关词，以后可以用于可视化或自动化文本分类。

### 统计方法

#### TF-IDF

统计方法最简单。他们计算关键字的统计数据并使用这些统计数据对它们进行评分。一些简单的统计方法是词频、词搭配和共现。有一些复杂的算法，如TF-IDF和YAKE。

TF-IDF是一种用于信息检索和文本挖掘的常用加权技术。用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
$$
TF-IDF = TF(t,d)*IDF(t)\\
tf_{i,j} = \frac{n_{i,j}}{\sum_kn_{k,j}}\\
idf_i=lg\frac{\mid D \mid}{\mid \{j:t_i \in d_j\}\mid}
$$


**词频**：$$n_{i,j}$$是该词在文件$$d_j$$中的出现次数，而分母是在文件$$d_j$$中所有字词的出现次数之和。

**逆文件频率**：是一个词语普遍重要性的度量。某一特定词语的IDF，可以由总文件数除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到。

因此TF-IDF 的想法是文档中出现频率更高的词不一定是最相关的。该算法偏爱在文本中频繁出现而在其他文本中不常见的术语。

#### YAKE（Yet Another Keyword Extractor）

YAKE是一种关键字提取方法，它利用单个文档的统计特征来提取关键字，需经历五个步骤 ：

1. 预处理和候选词识别---文本被分成句子、块和标记。文本被清理标记，停用词也会被识别。
2. 特征提取---计算文档中单词的五个统计特征
   1. 大小写---计算该术语在文本中出现大写或作为首字母缩略词的次数(与所有出现成比例)。重要的术语通常更频繁地出现大写。
   2. 词条位置---词条在文本中的中间位置。更接近开头的术语更重要。
   3. 词频归一化---测量文档中的平衡词频。
   4. 术语与上下文的相关性---衡量候选术语同时出现的不同术语数量。更重要的术语与较少不同的术语同时出现。
   5. 术语不同的句子---测量术语在不同句子中出现的次数。得分越高表示术语越重要。
3. 计算术语分数---上一步的特征与人造方程组合成一个单一的分数。
4. 生成n-gram并计算关键字分数---该算法识别所有的n-gram。n-gram中的单词必须属于同一块，并且不能以停用词开头或结尾。然后通过将每个n-gram的成员分数相乘并对其归一化，以减少n-gram长度的影响。停用词的处理方式有所不同，以尽量减少其影响。
5. 重复数据删除和排名---在最后一步算法删除相似的关键字。它保留了更相关的那个（分数较低的那个）。使用Levenshtien相似度、Jaro-Winkler相似度或序列匹配器计算相似度。最后，关键字列表根据它们的分数进行排序。

YAKE的优势在于它不依赖与外部语料库、文本文档的长度、语言或领域。与TF-IDF相比，它在单个文档的基础上提取关键字，并且不需要庞大的语料库。

下面是一个简单用法示例：

```python
import yake
text = "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning "\
"competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud "\
"Next conference in San Francisco this week, the official announcement could come as early as tomorrow. "\
"Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. "\
"Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, "\
"was founded by Goldbloom  and Ben Hamner in 2010. "\
"The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, "\
"it has managed to stay well ahead of them by focusing on its specific niche. "\
"The service is basically the de facto home for running data science and machine learning competitions. "\
"With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, "\
"it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow "\
"and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, "\
"Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. "\
"That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google "\
"will keep the service running - likely under its current name. While the acquisition is probably more about "\
"Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition "\
"and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can "\
"share this code on the platform (the company previously called them 'scripts'). "\
"Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with "\
"that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) "\
"since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, "\
"Google chief economist Hal Varian, Khosla Ventures and Yuri Milner "

kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords('text')

#specify parameters
#language = "en"
#max_ngram_size = 3
#deduplication_thresold = 0.9
#deduplication_algo = 'seqm'
#windowSize = 1
#numOfKeywords = 20

for kw in keywords:
  print(kw)
 
output:
('google', 0.026580863364597897)
('kaggle', 0.0289005976239829)
('ceo anthony goldbloom', 0.029946071606210194)
('san francisco', 0.048810837074825336)
('anthony goldbloom declined', 0.06176910090701819)
('google cloud platform', 0.06261974476422487)
('co-founder ceo anthony', 0.07357749587020043)
('acquiring kaggle', 0.08723571551039863)
('ceo anthony', 0.08915156857226395)
('anthony goldbloom', 0.09123482372372106)
('machine learning', 0.09147989238151344)
('kaggle co-founder ceo', 0.093805063905847)
('data', 0.097574333771058)
('google cloud', 0.10260128641464673)
('machine learning competitions', 0.10773000650607861)
('francisco this week', 0.11519915079240485)
('platform', 0.1183512305596321)
('conference in san', 0.12392066376108138)
('service', 0.12546743261462942)
('goldbloom', 0.14611408778815776)
```

### 基于图的方法

基于图的方法是从文档中生成相关术语。例如，图将文本中共同出现的术语连接起来。基于图的方法使用图排序，该方法考虑图的结构来对顶点重要性进行评分。最著名的基于图的方法之一是TextRank。步骤如下：

1. 带有词性(POS)标签的文本标记化和注释。
2. 词共现图构建---图中的顶点是带有选定PoS标签的词(作者仅选择名词和形容词即可获得最佳效果)。如果两个顶点出现在文本中N个单词的窗口内，则它们通过一条边相连（根据作者的实验，最佳表现N为2）。该图是无向和未加权的。
3. 根据TextRank的公式，初始化各节点的权重，然后迭代传播各节点的权重，直至收敛。
4. 对节点权重进行倒序排序，从而得到最重要的T个单词，作为候选关键词。
5. 由4得到最重要的T个单词，在原始文本中进行标记，若形成相邻词组，则组合成多词关键词。

### 深度学习之embed

#### **Embedrank**

>Embedrank: Unsupervised keyphrase extraction using sentence embeddings
>
>论文链接:[https://arxiv.org/abs/1801.04470](https://arxiv.org/abs/1801.04470)

**思路**：先利用POS tags 抽取候选短语，然后计算候选短语的embedding 和文章的embedding 的cosine si milarity，利用相似度将候选短语排序，得到关键短语。

**步骤**

1. 通过词性组合抽取候选短语：或者更多的形容词至少跟着一个名词；
2. 计算候选短语的embedding和doc embedding，计算doc embedding时进行降噪处理，只保留形容词+名词，doc计算时是按照保留的词生成doc 向量
3. 计算候选短语和doc的相似度得分进行排序，

>说明：针对上述方法提取出来的top短语，其中有很多表达相似的意思，为了增加短语的多样性，文章提出了embed rank++，采用信息检索中的MMR(最大边界相关性)增加多样性。

#### SIFRank

>论文：SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-Trained Language Model
>
>链接：[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954611](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954611)

**思路**：SIFrank 利用POS tags抽取NP作为keyphrase candidates，然后利用ELMO分别获得句子和候选短语的语义向量，最终通过计算两者的相似度进行打分。

**SIFrank++**：在SIFrank的基础上融入位置信息。

位置权重增加的方法:



$$
p(NP_i)=\frac{1}{p_i+\mu}
$$


作者采用候选词第一次出现的位置倒数作为位置权重，$$\mu$$s是超参数，并且对位置权重归一化。



$$
\hat p(NP_i)=softmax(p(NP_i))=\frac{exp(p(NP_I))}{\sum^N_{k=1}exp(p(NP_k))}
$$



最终的权重如下=位置权重$\times$候选词与文章的相似度



$$
SIFRank+(NP_i,d)=\hat p(NP_i) \times Sim(v_{NP_i},v_d)
$$



SIFRank的结构如下：

![1](/img/keyword-extract/1.png)

步骤：

1. 分词和词性标注。
2. 抽取名词以及名词块(正则模式匹配)作为候选关键词短语。
3. 将句子送如ELMO预训练模型获得每个token的表示。
4. 获得候选短语和句子的语义表示。
5. 计算候选词和文档的相似度，获得Top N输出。

实验结果表明，SIFRank在短文数据集Semeval和Insepec都是sota，sifrank++在duc2001(长文数据集)上是sota，说明位置偏移在长文数据集中很有效。





