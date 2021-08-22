---
layout:     post
title:      "如何训练一个BERT模型"
subtitle:   "BERT"
date:       2021-08-22 10:00:00
author:     "Wwt"
header-img: "img/pretrained_bert/bg.png"
catalog: true
tags:   
    - NLP
---

>本文参考自[How to Train a BERT Model From Scratch](https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6)，部分内容有删改。

我的许多文章都专注于BERT--这个模型出现并主导了自然语言处理（NLP）的世界，标志着语言模型的新时代。

对于之前没有使用过Transformers模型（例如BERT是什么）的人来说，这个过程看起来有点像这样：

- pip install transformers
- 初始化一个预训练的transformers模型--from_pretrained
- 在一些数据上测试
- 也许微调模型（再训练一些）

现在，这是一个很好的方法，但如果我们只这样做，我们就缺乏对创建自己的transformer模型理解。而且，如果我们不能创建自己的Transformer模型--我们必须依赖一个适合解决我们问题的预训练模型，但情况并非如此：

![1](/img/pretrained_bert/1.png)

因此，在本文中，我们探讨构建我们自己的Transformer模型必须采取的步骤-特别是BERT的进一步开发版本，称为RoBERT。

### 概述

这个过程有以下几个步骤，所以在我们深入研究之前，让我们总结一下我们需要做什么。概括来说，有四个关键部分：

- 获取数据
- 构建分词器
- 创建输入管道
- 训练模型

一旦我们完成了以上部分中的每一个，我们将使用我们构建的tokenizer和model-并将它们保存起来，以便我们可以像通常使用from_pretrained一样的方式使用它们。

### 获取数据

与任何机器学习项目一样，我们需要数据。在用于训练Transformer模型的数据方面，我们几乎可以使用任何文本数据。

而且，我们在互联网上有很多东西--那就是非结构化文本数据。

从互联网上抓取文本领域最大的数据集之一就是OSCAR数据集。OSCAR数据集拥有大量不同的语言-从头开始训练最清晰的用例之一是我们可以将BERT应用于一些不太常用的语言，例如泰卢固语或纳瓦霍语。

我的母语是英语-但我的女朋友是意大利人，所以她-劳拉，将评估我们讲意大利语的BERT模型- FiliBERTo的结果。

因此，要下载OSCAR数据集的意大利语部分，我们将使用HuggingFace的数据集库-我们可以使用pip install datasets安装它，然后我们下载OSCAR_IT:

```/python
from datasets import load_dataset
dataset = load_dataset('oscar', 'unshuffled_deduplicated_it')
```

我们来看看数据集对象。

![2](/img/pretrained_bert/2.png)

现在让我们以一种可以在构建分词器时使用的格式存储我们的数据。我们需要创建一组仅包含数据集文本特征的纯文本文件，我们将使用换行符\n拆分每个样本。

![3](/img/pretrained_bert/3.png)

在我们的data/text/Oscar_it目录中，我们会发现：

![4](/img/pretrained_bert/4.png)

### 构建分词器

接下来是分词器，当我们使用transformer，我们通常会加载一个分词器，连同各自的预训练模型-分词器是该过程中的关键组件。

我们在构建分词器时，我们将为它们提供我们所有的OSCAR数据，指定我们的词汇量大小（分词器的标记数）和一些特殊标记。

现在，RoBERT特殊token如下所示：

![5](/img/pretrained_bert/5.png)

因此，我们确保它们包含在tokenize的训练方法调用specical_tokens参数中。

![6](/img/pretrained_bert/6.png)

我们的分词器现在已经准备好，保存它的文件以备后用。

![7](/img/pretrained_bert/7.png)

现在我们有两个定文件定义了我们新的FiliBERTo分词器:

-  merges.txt -- 执行文本到标记的初始映射
- vocab.json -- 将token映射到token id

有了这些，我们可以继续初始化分词器，以便我们可以像使用其它任何from_pretrained分词器一样使用它。

#### 初始化分词器

我们首先使用之前构建的两个文件初始化分词器--简单的使用from_pretrained：

```python
from transformers import RobertTokenizer
tokenizer = RobertTokenizer.from_pretrained('filiberto',max_length=512)
```

现在我们的分词器已经准备好，我们可以尝试用它编码一些文本。编码时，我们通常使用两种方法，encode和encode_bacth。

![8](/img/pretrained_bert/8.png)

从编码对象标记中，我们将提取input_ids和attention_mask张量，在FiliBERTo中使用。

#### 创建输入管道

我们训练过程的输入管道是整个过程中比较复杂的部分。它包括我们获取原始OSCAR训练数据，对其进行转换，然后将其加载到准备进行训练的DataLoader中。

#### 准备数据

我们从一个示例开始，然后完成准备逻辑。

首先，我们需要打开我们的文件--我们之前保存为.txt的文件。我们根据换行符\n拆分每个文本，因为这表示单个样本。

```python
with open('../../data/text/oscar_it/text_0.txt','r',encoding='utf-8') as fp:
	lines = fp.read().split('\n')
```

然后使用tokenizer对数据进行编码--包括一些关键参数max_length，padding和truncation。

```python
batch=tokenizer(line,max_length=512,padding='max_length',truncation=True)
len(batch)
```

现在我们可以继续创建张量，我们将通过掩码语言模型建模来训练我们的模型。所以，我们需要三个张量：

- Input_ids：这里的token ids，其中约15%的token使用掩码mask进行掩码。

- attention_mask：是1和0的张量，标记真实/填充的位置，用于计算注意力。

- labels：这里的tokens是没有遮掩的。

  这里的attention_mask和标签张量是从批次中提取的。然后，input_ids张量需要更多的关注，对于这个张量，我们屏蔽了大约15%的标记，为它们分配标记ID 3 。

  ```python
  import torch
  labels = torch.tensor([x.ids for x in batch])
  mask = torch.tensor([x.attention_mask for x in batch])
  input_ids = labels.detach().clone()
  rand = torch.rand(input_ids.shape)
  mask_arr = (rand<.15)*(input_ids!=0)*(input_ids!=1)*(input_ids!=2)
  for i in range(input_ids.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    input_ids[i,selection] = 3
  ```

  在最终输出中，我们可以看到编码的input_ids张量的一部分，第一个token ID 是1--表示 [CLS] token，在张量周围有几个3个token ID--这些是我们新添加的[MASK]token。

  ##### 构建dataloader

  接下来，我们定义DataSet类--我们用它来将我们的三个张量初始化为Pytorch中torch.utils.data.Dataset对象。

  ![9](/img/pretrained_bert/9.png)

最后，我们的dataset被加载到Pytorch DataLoader对象中--在训练期间，我们将数据加载到我们的模型中。

### 训练模型

训练模型需要两样东西，我们的DataLoader和模型，我们以构造好DataLoader，但是没有模型。

#### 初始化模型

对于训练，我们需要一个原始的(未训练的)BERTLMHeadModel。要创建它，我们首先需要创建一个RoBERTa配置对象来描述我们想要用来初始化FiliBERTo的参数。

```python
from transformers import RobertCOnfig
config = RobertConfig(
  	vocab_size=30_522,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)
```

然后，我们使用语言建模(LM)导入和初始化RoBERTa模型。

```python
from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config)
```

#### 训练准备

在我们进入训练之前，我们需要设置一些东西。首先，我们设置GPU/CPU使用率。然后我们激活我们模型的训练模式。最后，初始化优化器。

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
# and move our model over to the selected device 
model.to(device) 
from transformers import AdamW 
# activate training mode 
model.train() 
# initialize optimizer 
optim = AdamW(model.parameters(), lr=1e-4)
```

#### 训练

最后--训练时间，我们按照Pytorch的方式进行训练。

```python
epochs = 2
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        model.save_pretrained('./filiberto')
```

如果我们继续使用TensorBoard，随着时间推移，我们会发现我们的损失--它看起来很有希望。

![10](/img/pretrained_bert/10.png)

#### 最终测试

现在是进行真正测试的时候了。我们建立了一个MLM管道--病情劳拉评估结果。

我们首先使用'fill-mask'参数初始化一个管道对象。然后像这样开始测试我们的模型。

```python
from transformers import pipeline
fill = pipeline('fill-mask', model='filiberto', tokenizer='filiberto')
Some weights of RobertaModel were not initialized from the model checkpoint at filiberto and are newly initialized: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
fill(f'ciao {fill.tokenizer.mask_token} va?')
```

![11](/img/pretrained_bert/11.png)

"ciao come va" 是正确答案。这和我的意大利语一样先进，接下来，让我们把它交给劳拉。使用更复杂的短语。

![12](/img/pretrained_bert/12.png)

最后，再来一句更难的短语，“cosa sarebbe successo se avessimo scelto un altro giorno？”--或者“如果我们选择另一天会发生什么？”

![13](/img/pretrained_bert/13.png)

总的来说，看起来我们的模型通过了劳拉的测试--我们现在有一个名为FiliBERTo的意大利语模型。

这就是从头开始训练BERT模型的训练！我们涵盖了很多方面，从获取和格式化我们的数据--一直到使用语言建模来训练我们的原始BERT模型。

### 参考

> [How to Train a BERT Model From Scratch](https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6)

