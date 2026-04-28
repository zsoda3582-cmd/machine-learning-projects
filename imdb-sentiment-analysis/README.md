# 🎬 IMDB 电影评论情感分析

# 项目简介
本项目基于 IMDB 电影评论数据，使用经典机器学习方法实现文本情感分类（正面 / 负面）。

通过对比不同模型的效果，加深对 NLP 文本分类流程的理解。

---

# 方法流程
经典机器学习文本分类 pipeline：
文本 → TF-IDF 向量化 → 模型训练 → 情感预测

使用模型：
- Logistic Regression（逻辑回归）
- Naive Bayes（朴素贝叶斯）
---

#📊 实验结果
模型效果对比：
| 模型 | 准确率 |
|------|--------|
| Logistic Regression | ≈ 0.90 |
| Naive Bayes | ≈ 0.86 |
| XGBoost | ≈ 0.79 |

---
# 核心理解
1️⃣ TF-IDF 的作用
将文本转为高维稀疏向量，表示词的重要程度。

2️⃣ Logistic Regression
- 适合高维稀疏数据
- 能学习词与情感之间的权重关系
- 表现最稳定

3️⃣ Naive Bayes
- 假设词之间相互独立
- 计算简单，速度快
- 效果略低于 LR

4️⃣ XGBoost
- 不适合稀疏文本特征
- 在本任务中表现较差

---
# 功能说明
✔ 模型训练（Logistic Regression）  
✔ 模型对比（LR vs NB vs XGBoost）  
✔ 命令行交互预测（输入一句话 → 输出情感 + 概率）  
✔ 支持模型切换（LR / NB）

---
# 项目收获
- 理解文本分类基本流程（TF-IDF + 模型）
- 掌握 Logistic Regression 在 NLP 中的应用
- 初步接触 Naive Bayes 模型
- 学会模型效果对比分析
- 完成一个完整可运行的小项目
