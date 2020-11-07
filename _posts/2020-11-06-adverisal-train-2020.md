---
layout:     post
title:      "闲话NLP中的对抗训练"
subtitle:   "对抗训练"
date:       2020-11-06 10:00:00
author:     "Wwt"
header-img: "img/advtraining/bg.png"
catalog: true
tags:   
    - 对抗训练
    - NLP
---
### 简介

提到“对抗”，相信大多数人的第一反应都是CV中的对抗生成网络（GAN），殊不知，其实对抗也可以作为一种**防御机制**，并且经过简单的修改，便能用在NLP任务上，提高模型的泛化能力。关键是，对抗训练可以写成一个插件的形式，用几行代码就可以在训练中自由地调用，**简单有效，使用成本低**。目前网上对NLP中对抗训练有一些介绍，笔者在这篇博客中对自己在比赛中使用了对抗训练技巧做一下记录。

>一开始打比赛，看到大佬们都说使用对抗训练提升了一丢丢的效果，让我仿佛看到了炼丹的法宝，实际自己上手一用，效果不增反降。简直是反向操作，心里苦啊。
>
>本篇博客参考自 [功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)

首先来了解一下什么是“对抗训练”

> **对抗训练是一种引入噪声的训练方式，可以对参数进行正则化，提升模型鲁棒性和泛化能力。**

什么样的样本才是最好的对抗样本呢？对抗样本一般需要具备两个特点：

>- 相对于原始输入，所添加的扰动是微小的；
>- 能使模型犯错。

### 对抗训练的基本概念

GAN之父Ian Goodfellow在15年的ICLR中第一次提出了对抗训练这个概念，简而言之，就是在原始输入样本$x$加上一个扰动$r_adv$，得到对抗样本后，用其进行训练。也就是说，问题可以被抽象成这么一个模型：


$$
min_\theta-log P(y\mid x+r_{adv};\theta)
$$


其中，$y$为gold label，$\theta$为模型参数。那么扰动要如何计算呢？Goodfellow认为，**神经网络由于其线性的特点，很容易受到扰动的攻击**。

于是，他提出了Fast Gradient Sign Method(FGSM)，来计算输入样本的扰动。扰动可以被定义为:


$$
r_{adv} = \epsilon *sgn(\nabla_xL(\theta,x,y))
$$


其中，$sgn$为符号函数，$L$为损失函数。Goodfellow发现，令$\epsilon=0.25$,用这个扰动能给一个单层分类起造成99.9%的错误率。看似这个扰动的发现有点拍脑门，但是仔细想想，其实这个扰动计算的思想可以理解为：将输入样本向着损失上升的方向再进一步，得到的对抗样本就能造成更大的损失，提高模型的错误率。回想我们上一节提到的对抗样本的两个要求，FGSM刚好可以完美地解决。

Goodfellow还总结了对抗训练的两个作用：

>- 提高模型应对恶意样本时的鲁棒性；
>- 作为一种regularization，减少过拟合，提高泛化能力。

### Min-Max公式

对抗训练的理论部分被阐述得还是比较intuitive，Madry在2018年的LCLR中总结了之前的工作，并从优化的角度，将问题重新定义成了一个找鞍点的问题，也就是大名鼎鼎的MIin-Max公式：


$$
min_\theta E_{(x,y)}\sim D[max_{r_{adv}\in S}L(\theta,x+r_{adv},y)]
$$


该公式分为两个部分，一个是内部损失函数的最大化，一个是外部经验风险的最小化。

>1. 内部max是为了找到worst-case的扰动，也就是攻击，其中，$L$为损失函数，$S$为扰动的范围空间。
>2. 外部min是为了基于该攻击方式，找到最鲁棒的模型参数，也就是防御，其中$D$是输入样本的分布。
>

上述公式简单清晰地定义了对抗样本攻防“矛与盾”的两个问题：如何构造足够强的对抗样本？以及，如何使模型变得刀枪不入？剩下的，就是如何求解的问题了。

### 从CV到NLP

以上提到的一些工作都还是停留在CV的领域，那么问题来了，可否将对抗训练迁移到NLP上呢？答案是肯定的，但是，我们得考虑这么几个问题：

首先，cv任务的输入是连续的RGB值，而NLP问题中，输入是离散的单词序列，一般以one-hot vector的形式呈现，如果直接在raw text 上进行扰动，那么扰动的大小和方向可能都没什么意义。Goodfellow在17年的ICLR中提出了可以在连续的embedding上做了扰动。

乍一思考，觉得这个方案似乎特别完美，。然而，对比图像领域中直接在原始输入加扰动的做法，在embedding上加扰动会带来这么一个问题：这个被构造出来的“对抗样本”并不能对应到某个单词上，因此，反过来在inference的时候，对手也没有办法通过修改原始输入得到这样的对抗样本。在CV任务，根据经验性的结论，对抗训练往往使得模型在非对抗样本上表现变差，然而神奇的是，在NLP任务重，模型的泛化能力反而变强了。

因此，在NLP任务重，**对抗训练的角色不再是为了防御基于梯度的恶意攻击，反而更多的是作为一种正则化，提高模型的泛化能力**。

用一句话形容对抗训练的思路，就是**在输入上进行梯度上升(增大loss)，在参数上进行梯度下降（减小loss）**。由于输入会进行embedding lookup，所以实际的做法是在embedding table进行梯度上升。

### NLP中的两种对抗训练+Pytorch实现

#### a.Fast Gradient Method（FGM）

上面我们提到，Goodfellow在15年的ICLR中提出了Fast Gradient Sign Method(FGSM)，随后，在17年的ICLR中，Goodfellow对FGSM中计算扰动的部分做了一点简单的修改。假设输入的文本序列embedding vectors$$[v_1,v_2,...,v_T]$$为$x$，embedding 的扰动为：


$$
r_{adv} =\epsilon * g/\mid\mid g \mid\mid\\
g=\nabla_xL(\theta,x,y)
$$
实际上就是取消了符号函数，用二范式做了一个scale，需要注意的是：这里的norm计算的是，每个样本的输入序列中出现过的词组成的矩阵的梯度norm。为了实现插件式的调用，笔者将一个batch抽象成一个样本，一个batch统一用一个norm，由于本来norm也只是一个scale的作用，影响不大。笔者的实现如下：

```python
import torch
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

需要使用对抗训练的时候，只需要添加五行代码：

```python
# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    # 对抗训练
    fgm.attack() # 在embedding上添加对抗扰动
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
```



#### **b. Projected Gradient Descent（PGD）**

内部max的过程，本质上是一个非凹的约束优化问题，FGM解决的思路其实就是梯度上升，**那么FGM简单粗暴的“一步到位”，是不是有可能并不能走到约束内的最优点呢？**当然是有可能的。于是，一个很intuitive的改进诞生了：Madry在18年的ICLR中，提出了用Projected Gradient Descent（PGD）的方法，简单的说，就是**“小步走，多走几步”**，如果走出了扰动半径为$\epsilon$的空间，就映射回“球面”上，以保证扰动不要过大：


$$
x_{t+1}=\Pi_{x+S}(x_t+\alpha g(x_t)/\mid\mid g(x_t)\mid\mid_2)\\
g(x_t) = \nabla_xL(\theta,x_t,y)\
$$
其中
$$
S=r \in \R^d: \mid\mid r \mid\mid_2 \leq \epsilon
$$
为扰动的约束空间，$\alpha$为小步的步长。

```python
import torch
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
```

使用的时候，要麻烦一点：

```python
pgd = PGD(model)
K = 3
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    pgd.backup_grad()
    # 对抗训练
    for t in range(K):
        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K-1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
```

### 总结

上面简单地介绍了NLP中对抗训练概念以及常用的两种对抗训练方式，在一些NLP任务取得了一些性能的提升。不过，根据我们使用的经验来看，是否有效有时也取决于数据集。毕竟炼丹，真的很奇妙。