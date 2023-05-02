---
layout:     post
title:      Alpaca-lora开源实现部署
subtitle:   "Alpaca"
date:       2023-04-30 10:00:00
author:     "Wwt"
header-img: "img/alpaca-lora/bg.png"
catalog: true
tags:   
    - NLP
---

> 本文参考自[# Alpaca-Lora (羊驼-Lora): 轻量级 ChatGPT 的开源实现（对标 Standford Alpaca](https://zhuanlan.zhihu.com/p/615646636)，部分有删改

### 总览

本文介绍Alpaca-Lora(羊驼-Lora)，可以认为是ChatGPT轻量级的开源版本，它使用Lora(Low rank Adaption)技术在Meta 的LLMA 7B模型上进行微调。只需要训练很小一部分参数就可以获得媲美Standford Alpaca模型的效果。本文主要**介绍它在本地安装使用的方法。**

### Alpaca原理介绍

Alpaca的介绍在[Alpaca: A Strong, Replicable Instruction-Following Model]([Stanford CRFM](https://crfm.stanford.edu/2023/03/13/alpaca.html)),模型的训练流程基本可以用下图来概括。

![截屏2023-04-23 22.24.11.png](/img/alpaca-lora/1.png)

它使用了52K个instruction -following examples 来微调Meta的大语言模型LLaMA 7B(Meta 开放了模型权重以及inference代码，详见 https://github.com/facebookresearch/llama ) ,从而生成了Alpaca 7B。

但是这52K个instruction-following examples是如何生成的呢？Alpaca 团队使用了[https://github.com/yizhongw/self-instruct]()。不得不说，这种方法很巧妙，很像蒸馏训练。将OpenAI性能完备的模型作为Teacher,来指导参数更少的Alpaca模型进行训练，大幅降低了训练成本。其中调用Open API的成本不到500美刀，另外微调7B参数的LLaMA模型，使用云服务商提供的8块80GB A100显卡，训练3小时，消费不到100美刀。因此整体成本是小于600美刀。

### LoRA简要介绍

关于Alpaca-lora和stanford Alpaca模型的区别，先入为主的印象是,standford alpaca是在LLaMA整个模型上微调，而Alpaca-Lora则是利用Lora技术（LoRA：Low-Rank Adaption of Large Language Models），在冻结原模型的LLaMA参数的情况下，通过往模型中加入额外的网络层，并只训练这些新增的网络层参数。由于这些新增参数数量较少，这样不仅finetune的成本明显下降，还能获得和全模型微调类似的效果。

![](/img/alpaca-lora/2.png)

蓝色模块是原模型，橙色模块是新增网络层，通过控制参数r的大小，可以有效减少新增网络层的参数。

### 各类资源

[Alpaca-Lora ](https://link.zhihu.com/?target=https%3A//github.com/tloen/alpaca-lora)

[LLaMA-7B-HF](https://link.zhihu.com/?target=https%3A//huggingface.co/decapoda-research/llama-7b-hf)

[Lora参数地址](https://link.zhihu.com/?target=https%3A//huggingface.co/tloen/alpaca-lora-7b)

[如何优雅的下载huggingface transformers模型](https://zhuanlan.zhihu.com/p/475260268)

### 本地安装

README 文件中说明使用pip install -r requirements.txt就OK了。另外我们还发布了一个脚本,用于在基础模型和LoRA上下载和推理，以及生成LoRA权重本身。为了便宜且高效地进行微调，我们使用Hugging Face的PEFT和bitsandbytes。

### 运行generate.py

正常情况下，如果有超过8G的GPU显存，并且网络之类的都相当好的话，那么直接运行python generate.py就能成功。

但是实际操作过程中，会遇到太多不正常的情况。

首先是模型参数的下载，包括LLaMA-7B-hf大模型以及LoRA参数，下载报HTTP Requests之类的错误。

我参考上述提到的[如何优雅的下载huggingface-transformers模型](https://zhuanlan.zhihu.com/p/475260268)，安装huggingface_hub 进行模型下载，速度较快，执行如下命令下载模型：

> ```bash
> >>>from huggingface_hub import snapshot_download
> >>> snapshot_download(repo_id="decapoda-research/llama-7b-hf")
> ```

结果如下：

![截屏2023-04-23 23.06.53.png](/img/alpaca-lora/3.png)

模型下载成功后，终端会输出模型的保存地址，可以使用stat -Lc "%n %s" *，大致看一下各个文件是否有缺失，和Huggingface 上的模型大小简单对比一下。

别忘了LoRA模型，执行`snapshot_download(repo_id="tloen/alpaca-lora-7b")` 下载 Lora 参数。

所有参数下载完成以后，继续运行generate.py，一处断言代码会报错，直接注释掉即可。

要想成功运行，需要至少**8GB的GPU显存**。

### 小结

本文介绍了ChatGPT轻量级的开源版本Alpaca-Lora（羊驼-lora），使用Lora(Low-Rank Adapation)技术在Meta 的LLaMA 7B模型上微调，只需要训练很小一部分参数就可以获得媲美Stanardford Alpaca模型的效果。
