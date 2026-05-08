## News Classification NLP Project（新闻文本分类项目）
# 项目简介
本项目基于 AG News 数据集， 使用 TF-IDF + 机器学习模型 实现新闻文本多分类任务。 
分类类别包括： 
- World
- Sports
- Business
- Sci/Tech

# 项目流程
1. 数据读取
2. 文本拼接
3. TF-IDF 向量化
4. 模型训练
5. 多模型对比
6. confusion matrix 分析
7. heatmap 可视化

# 使用模型
本项目对比了三种经典 NLP 模型： 
- Logistic Regression
- Multinomial Naive Bayes
- LinearSVC

# 实验结果
| 模型 | Accuracy | 特点 | 
|------|------|------| 
| Logistic Regression | 0.92 | 稳定平衡 | 
| Naive Bayes | 0.90 | 训练快，但误分类较多 | 
| LinearSVC | 0.93 | 效果最好 |

<img width="892" height="839" alt="image" src="https://github.com/user-attachments/assets/93844284-6ed6-4783-8329-2d0ec7fe7c5e" />
<img width="893" height="835" alt="image (1)" src="https://github.com/user-attachments/assets/39774bc7-c454-489e-8ece-275fab7c9388" />
<img width="918" height="842" alt="image (2)" src="https://github.com/user-attachments/assets/96f00354-9ca9-4502-9b82-6bb467d304d1" />

# 错误分析
通过 confusion matrix 发现： 
- Sport 类别最容易分类
- Business 与 Sci/Tech 最容易混淆
原因： 很多科技新闻本身也带有商业属性， 例如 Apple、Microsoft、AI company 等新闻。

# 项目收获
通过本项目，我学习了： 
- TF-IDF 文本向量化
- 多分类任务
- confusion matrix
- heatmap 可视化
- 不同模型的效果对比
- 文本分类中的错误分析
