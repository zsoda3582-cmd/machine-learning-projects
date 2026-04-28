# =========================
# 1️⃣ 读取数据
# =========================
import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")
print(df.shape)
print(df.head())
print("-"*50)

# =========================
# 2️⃣ 列名标准化
# IMDB数据一般是：
# review + sentiment
# 我们统一改成：
# text + label
# =========================
df = df.rename(columns={
    "review" : "text",
    "sentiment" : "label"
})
print(df.head())
print(df["label"].value_counts())
print("-"*50)

# =========================
# 3️⃣ label 转成 0/1
# positive → 1
# negative → 0
# =========================
df["label"] = df["label"].map({
    "positive":1,
    "negative":0
})
print(df.head())
print(df["label"].value_counts())
print("-"*50)

# =========================
# 4️⃣ 文本特征提取：TF-IDF
# 把文本转成机器学习模型能理解的数字特征
# ngram_range=(1,2)：同时考虑单个词和连续两个词
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer
#TF-IDF：TF（词频）——一个词在这条评论里出现越多 → 越重要；IDF（逆文档频率）——一个词在所有评论里越常见 → 越不重要；TF-IDF = 常在本句出现 × 不常在全局出现
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features = 5000
)
x = vectorizer.fit_transform(df["text"])
print(x.shape)
print(vectorizer.get_feature_names_out())
print("-"*50)

#建模
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
x_train,x_test,y_train,y_test = train_test_split(
    x,df["label"],test_size = 0.25,random_state=42
)
model = LogisticRegression()
nb_model = MultinomialNB()
xgb_model = XGBClassifier(
    n_estimators = 50,
    max_depth = 3,
    learning_rate = 0.1,
    eval_metric = "logloss",
    random_state = 42,
    n_jobs = -1
)
model.fit(x_train,y_train)
nb_model.fit(x_train,y_train)
xgb_model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_nb = nb_model.predict(x_test)
y_pred_xgb = xgb_model.predict(x_test)
print("LogisticRegression结果：")
print(classification_report(y_test, y_pred))
print("Naive Bayes结果：")
print(classification_report(y_test, y_pred_nb))
print("xgboost结果：")
print(classification_report(y_test, y_pred_xgb))

"""
运行结果分析：
Naive Bayes是一种朴素贝叶斯模型， 假设词之间独立，计算简单但无法建模词之间的关系。
而 Logistic Regression 可以利用特征组合（如 n-gram），
因此在情感分类中表现更稳定。
"""

# =========================
# 简单交互预测
# =========================
def predict_sentiment(text,model):
    x_new = vectorizer.transform([text])
    pred = model.predict(x_new)
    prob = model.predict_proba(x_new)
    if pred[0] == 1:
        print(f"正面😀，概率：{prob[0][1]:.2f}")
    else:
        print(f"负面😟，概率：{prob[0][0]:.2f}")

while True:
    text = input("请输入一句话（输入exit退出）")
    if text == "exit":
        break
    model_choice = input("请选择模型（LR/NB）：")
    if model_choice == "LR":
        result = predict_sentiment(text,model)
    elif model_choice == "NB":
        result = predict_sentiment(text,nb_model)
    else:
        print("请选择模型（LR/NB）：")
    