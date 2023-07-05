# keras-bert-poetry-generator
代码来源于
https://github.com/AaronJny/DeepLearningExamples/blob/master/keras-bert-poetry-generator/README.MD

BERT模型使用的是[苏剑林](https://github.com/bojone)大佬封装的[bert4keras](https://github.com/bojone/bert4keras)，开发过程中参考了bert4keras的[demo](https://github.com/bojone/bert4keras/tree/master/examples)。

模型参数使用的是Google官方在中文语料上的预训练权重[BERT-Base, Chinese](https://github.com/google-research/bert#pre-trained-models)，下载地址为[https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)


## 1. 如何训练模型？

- clone项目到本地。
- 下载Google的预训练权重。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`keras-bert-poetry-generator`路径下，执行命令`python3 train.py`。

## 2. 如何使用训练好的模型生成古诗？

- clone项目到本地。
- 下载Google的预训练权重。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`keras-bert-poetry-generator`路径下，执行命令`python3 eval.py`。

## NEZHA
In ```model.py```,```line 11```，set ```model='NEZHA'``` 