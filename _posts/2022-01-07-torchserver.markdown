---
layout:     post
title:      TorchServe部署transformers model
subtitle:   "TorchServe"
date:       2022-01-07 10:00:00
author:     "Wwt"
header-img: "img/torchserve/bg.png"
catalog: true
tags:   
    - NLP
---

> 本篇博客来源于[如何部署Pytorch模型](https://zhuanlan.zhihu.com/p/344364948)，有部分改动。

### 什么是TorchServe?

使用Pytorch框架训练好了模型，如何部署到生产环境提供服务呢？

有Web开发经验的小伙伴可能首先想到的是用HTTP框架（例如Flask）进行简单的封装，做一个简单的HTTP接口来对外提供服务。但既然是生产，那还是免不了考虑性能、扩展性、可运维性等因素。另外，做好这个服务还要求开发人员具备良好的Web后端开发技术栈。好在PyTorch已经给我们提供了一套统一的标准来实现这些，这个也是这篇博客介绍的开源工具：TorchServe。

TorchServe是Pytorch官方开发的开源工具，源码地址[GitHub - pytorch/serve: Model Serving on PyTorch](https://github.com/pytorch/serve)。

官方给出的描述是：

> A flexible and easy to use tool for serving PyTorch models

从描述中就可以知道TorchServe是用来部署Pytorch模型的，而它的特点是**可扩展性**和**易用性**

### 为什么用它?

理由很多，这里列出笔者认为比较重要的几点。

1. Pytorch 官方出品，大厂出品无脑跟风就对了，不然为什么这么多人用Tensorflow(😁)

2. 隐藏工程实现细节，对数据科学家友好。
   
   使用Pytorch的工程师、科学家可能不知道如何实现工程相关的功能，例如RPC 、RESTful     API ,但是他们一定懂得预处理(preporcessing)和Pytorch模型。而知道这些就足够了，工     程上的事情，交给torchserve来处理，而且它还做的不错。人生苦短，不要重复造轮子。

3. 制定了标准。
   
   由于TorchServe将系统工程和模型分开了，所以模型接入到TorchServe需要一套统一的标准，而TorchServe将这套标准制定得很简单。我们甚至可以直接将预训练或测试的代码一部分截取下来，稍加改动即可完成接入操作。

除了以上还有很多其它的使用Torchserve的理由，Torchserve为我们提供了丰富的功能，例如日志，多进程，动态注册模型等。

下面是torchserve的完整架构图

![截屏2022-01-07 下午1.46.23.png](/img/torchserve/1.png)

### 如何使用

#### 使用docker安装

安装torchserve最好的方法是使用docker。你只需把镜像拉下来，可以使用以下命令保存最新的镜像。

> docker pull pytorch/torchserve:latest

启动容器

> docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/Huggingface_Transformers:/home/model-server/Huggingface_Transformers pytorch/torchserve:latest

#### 打包模型

使用torch-modelparchiver命令来打包模型(该命令在安装完Torchserve后会自动获得)。

下面以transformer框架bert模型为例，进行torchserve部署。

你需要准备两到三个文件：

1. checkpoint.pth.tar
   
   从命名就应该知道，这就是我们在训练过程中通过torch.save获得的模型权重文件，注意该文件内容只能包含模型的权重
   
   ![截屏2022-01-07 下午2.01.41.png](/img/torchserve/2.png)

2. model.py
   
   该文件应该包含单个模型的类，该模型类应该可以使用load_state_dict来成功加载checkpoint.pth.tar提供的权重。
   
   ```
   class BertCrfForNer(BertPreTrainedModel):
       def __init__(self, config):
           super(BertCrfForNer, self).__init__(config)
           self.bert = BertModel(config)
           self.dropout = nn.Dropout(config.hidden_dropout_prob)
           self.classifier = nn.Linear(config.hidden_size, config.num_labels)
           self.crf = CRF(num_tags=config.num_labels, batch_first=True)
           self.init_weights()
   
       def forward(self, input_ids, attention_mask=None):
           outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask)
           sequence_output = outputs[0]
           sequence_output = self.dropout(sequence_output)
           logits = self.classifier(sequence_output)
           outputs = (logits,)
           return outputs
   ```

3. handler.py（可选）
   
   如果希望加入一些自定义的preporcessing和postprocessing，可以加入该文件。比如，基于transformer框架的预训练语言模型需要通过from_pretrained函数来加载，我们需要改写handler下的__load__pickled__model函数
   
   ```
   def _load_pickled_model(self, model_dir, model_file):
           """
           Loads the pickle file from the given model path.
           Args:
               model_dir (str): Points to the location of the model artefacts.
               model_file (.py): the file which contains the model class.
               model_pt_path (str): points to the location of the model pickle file.
           Raises:
               RuntimeError: It raises this error when the model.py file is missing.
               ValueError: Raises value error when there is more than one class in the label,
                           since the mapping supports only one label per class.
           Returns:
               serialized model file: Returns the pickled pytorch model file
           """
           model_def_path = os.path.join(model_dir, model_file)
           if not os.path.isfile(model_def_path):
               raise RuntimeError("Missing the model.py file")
   
           module = importlib.import_module(model_file.split(".")[0])
           model_class_definitions = list_classes_from_module(module)
   
           logger.info("one class as model definition. {}".format(
                       model_class_definitions[0]))
   
           bertconfig = BertConfig.from_pretrained(model_dir,num_labels = 22)
           # logger.info('config:{}'.format(bertconfig))
   
           model_class = model_class_definitions[0]
           model = model_class.from_pretrained(model_dir,config = bertconfig)
           model.to(self.device)
           return model
   ```

上述handler文件中包含了预处理（将text转换成模型接收的张量输入），推理（将preprocess得到的张量，输入到模型中，获得概率输出），后处理（我们需要想客户返回一些内容）。torchserve总是返回一个数组。handler会自动打开一个.json文件带有index->label的映射，并将其存储到self.mapping中，我们可以将预测得到的id转换成实际标签。

准备好以上几个文件后，我们就可以使用torch-model-archiver打包，

> torch-model-archiver --model-name BERTNER --version 1.0 --model-file /home/model-server/examples/ner_model.py --serialized-file /home/model-server/examples/pytorch_model.bin  --export-path /home/model-server/model-store/ --extra-files /home/model-server/examples/ner/index_label.json --handler /home/model-server/examples/ner_handler.py

这里的参数都比较容易理解，但注意--model-name 参数我们可以取一个有意义的名称，该参数会影响到我们以后的调用服务的URL链接。

执行上述命令后，我们会得到一个mar的文件，这就是我们打包好的文件。

#### 注册模型

创建一个目录，名称为model-store，将第一步打包好的.mar复制到这个目录中，然后我们就可以启动Torchserve服务器程序了。

> torchserve --start --model-store model_store --models my_tc=BERTNER.mar --ncs

然后进行推理

> curl -X POST http://127.0.0.1:8080/predictions/my_tc -T Seq_classification_artifacts/sample_text_captum_input.txt 

服务端接收请求后，然后返回结果。整个流程结束。

### 总结

本篇博客简单介绍了使用docker安装torchserve步骤和handlers文件的功能，并且以transformers的bert模型为例，讲述了如何自定义该文件来实现加载预训练模型，然后，将模型打包生成mar 文件，最后使用docker提供模型服务。
