# 📩 Spam Classification 垃圾短信分类

## 📌 项目简介

本项目基于 SMS Spam 数据集，使用 TF-IDF + Logistic Regression 完成垃圾短信分类任务。

项目重点不只是训练模型，而是通过调整分类阈值（threshold），观察 precision 和 recall 的变化，理解模型在不同业务场景下的取舍。

---

## 📊 数据集

- 数据文件：spam.csv
- 标签：
  - ham：正常短信
  - spam：垃圾短信

---

## 🔧 方法流程

整体流程：

```text
短信文本 → 文本清洗 → TF-IDF 向量化 → Logistic Regression → threshold 调整
文本清洗
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

作用：

将文本统一转为小写
去除数字、标点和特殊符号
保留英文单词，减少噪声
🤖 使用模型

本项目使用：

LogisticRegression()

文本向量化方法：

TfidfVectorizer()
📈 Threshold 调整实验

默认情况下，模型通常使用：

probability > 0.5 → spam

本项目手动测试多个 threshold：

for t in [0.7, 0.5, 0.4, 0.3, 0.2, 0.1]:
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob > t).astype(int)
📊 实验结果
Threshold	Spam Precision	Spam Recall
0.7	0.99	0.54
0.5	0.99	0.73
0.4	0.98	0.81
0.3	0.96	0.85
0.2	0.94	0.91
0.1	0.67	0.97
<img width="960" height="810" alt="image" src="https://github.com/user-attachments/assets/ad2e2830-aa8c-487a-b9e3-3938bc9b3863" />

🔍 结果分析

随着 threshold 降低：

spam recall 提升：模型能抓到更多垃圾短信
spam precision 下降：模型更容易误判正常短信

说明 threshold 可以控制模型的“保守 / 激进”程度。

📌 业务理解

不同场景下，threshold 的选择不同：

场景	优先指标	策略
邮件 / 短信垃圾过滤	Precision	避免误伤正常信息
银行反诈骗	Recall	尽量不要漏掉风险
医疗疾病筛查	Recall	尽量不要漏诊
普通分类任务	平衡	可选择 0.2 ~ 0.3
📉 可视化

本项目绘制了 precision / recall 随 threshold 变化的曲线，用于观察二者之间的 trade-off。

plt.plot(thresholds, recalls, label="recall")
plt.plot(thresholds, precisions, label="precision")
plt.gca().invert_xaxis()
plt.legend()
plt.grid()
plt.show()
🚀 项目收获

通过本项目，我完成了：

文本分类基础流程
文本清洗
TF-IDF 向量化
Logistic Regression 建模
classification_report 结果分析
threshold 调整
precision / recall trade-off 理解
📌 总结

本项目展示了一个垃圾短信分类模型的完整流程。
通过 threshold 调整，可以根据不同业务目标控制模型行为：

想抓得准：提高 threshold
想抓得多：降低 threshold
