## IMDB LSTM Sentiment Analysis（电影评论情感分析项目）
# 项目简介
本项目基于 IMDB Dataset 数据集，
使用：
- TensorFlow
- Keras
- LSTM

实现电影评论情感二分类任务，情感类别包括：
- Positive
- Negative

# 项目流程
1. 数据读取
2. 标签编码（positive/negative → 1/0）
3. 文本分词（Tokenizer）
4. 文本转数字序列（sequence）
5. padding 补齐长度
6. 构建 LSTM 模型
7. 模型训练
8. 模型测试
9. 输入评论预测情感

# 使用模型
本项目使用：
- Embedding
- LSTM
- Dense(sigmoid)
构建深度学习文本分类模型。

模型结构：Embedding → LSTM → Dense

# 核心知识点
1. Tokenizer：将单词转换成数字编号，eg:"movie is good" → [12, 7, 35]
2. Sequence:将文本转换成对应的数字序列（LSTM 无法直接读取字符串，只能读取数字）
3. Padding：由于每条评论长度不同，需要统一长度。本项目设置maxlen = 200，不足补 0，超出则截断。

# 模型训练
history = model.fit(
    x_train_pad,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)
参数含义：
- epochs：训练轮数
- batch_size：每批训练数据量
- validation_split：划分验证集

# 实验结果
测试集准确率≈ 0.85 ~ 0.87
现象：
对长评论效果较好/对超短句子预测不稳定/存在部分误分类

例如："I love this movie"被错误预测为 negative。

原因：
- 训练数据主要是长影评，
- 短句信息量不足。

# 项目收获
通过本项目，我学习了：
- TensorFlow 基础使用
- Keras Sequential 模型
- NLP 深度学习流程Tokenizer 与 sequence
- padding 补齐
- LSTM 文本处理
- 深度学习模型训练方式
- sentiment analysis 情感分析

# 当前问题
目前模型仍存在：
短文本预测不稳定
部分 positive 被误判
泛化能力不足

后续可继续优化：
增加 epochs
数据清洗
使用 Bidirectional LSTM
使用预训练词向量
使用 Transformer/BERT
