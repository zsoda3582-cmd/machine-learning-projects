## Word2Vec Text Analysis
# 项目简介
本项目从零开始学习 Word2Vec 与 Embedding 的核心思想。
通过手动构造 Skip-Gram 训练数据，理解：
- 单词如何编号
- Embedding 层本质
- 词向量如何训练
- Word2Vec 如何通过上下文学习语义关系

最终使用 gensim 训练一个简单 Word2Vec 模型，并观察词语相似度。

# 项目内容
1. 文本预处理
- sentence.split()
- 构建 word_to_index
- 单词编号化

2. 手动构造 Skip-Gram 数据
理解：
- center word
- context word
- window size
手动生成：(center_word, context_word) 训练样本。

3. Embedding 理解
学习：nn.Embedding(vocab_size, embedding_dim)
理解：
- Embedding 本质是“可训练向量表”
- 输入编号 → 输出词向量
- shape 的含义

4. 句子向量化
理解 NLP 常见输入 shape：(batch_size, seq_len, embedding_dim)，并理解 LSTM 实际处理的是词向量序列。

5. Word2Vec 训练
使用：gensim.models.Word2Vec 训练词向量。

学习参数：
- vector_size
- window
- sg
- epochs

6. 词向量相似度
使用：
- model.wv.similarity()
- model.wv.most_similar()

观察：
- movie 与 film 的相似度
- 词向量如何学习语义关系

# 项目收获
- 初步理解 Word2Vec 核心思想
- 理解 Embedding 的本质
- 理解 NLP 深度学习输入数据形式
- 理解“词义来自上下文”
- 初步进入深度学习 NLP 领域
