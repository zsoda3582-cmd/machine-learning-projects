## AG News LSTM Text Classification

# 项目简介
这是一个基于 PyTorch 的 NLP 新闻分类项目，使用 LSTM 对 AG News 数据集进行四分类任务。

# 项目完整实现了：
- 文本预处理
- 构建词表
- 文本转 token ids
- padding 统一长度
- Tensor 转换
- Embedding 层
- LSTM 网络
- Linear 分类层
- CrossEntropyLoss
- Adam 优化器
- mini-batch 训练
- 测试集评估

# 最终测试集准确率约：89%

# 数据集
AG News 数据集包含四类新闻：
- World
- Sports
- Business
- Sci/Tech

# 项目流程
文本
↓
分词 split
↓
构建词表 word_to_index
↓
文本转 token ids
↓
padding 统一长度
↓
Tensor 化
↓
Embedding
↓
LSTM
↓
Linear 分类
↓
Softmax 概率
↓
CrossEntropyLoss
↓
反向传播
↓
参数更新

# 模型结构Embedding -> LSTM -> Linear

# 参数设置：
- embedding_dim = 64
- hidden_size = 128
- batch_size = 64
- optimizer = Adam
- loss = CrossEntropyLoss

# 训练结果
训练过程中 loss 持续下降：
- epoch1 loss: 1026
- epoch2 loss: 500
- epoch3 loss: 355

测试集准确率：test_accuracy ≈ 0.89

# 项目收获
通过这个项目，学习了：
- NLP 文本数字化流程
- token / padding 的作用
- Embedding 的本质
- LSTM 如何处理序列数据
- logits / softmax / loss 的关系
- PyTorch 神经网络训练流程
- mini-batch 训练方式
- 反向传播与参数更新机制
