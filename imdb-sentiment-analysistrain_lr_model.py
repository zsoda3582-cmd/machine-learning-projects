# =========================
# 1️⃣ 读取数据
# =========================
import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")
print(df.shape)
print(df.head())
print("-"*50)

# texts = [
#     "I love this movie",
#     "This film is amazing",
#     "So good and enjoyable",
#     "I really liked it",
#     "Fantastic experience",
#     "Absolutely wonderful",
    
#     "I hate this movie",
#     "This film is terrible",
#     "Very bad experience",
#     "I dislike it",
#     "Awful and boring",
#     "Not good at all"
# ]

# labels = [1,1,1,1,1,1, 0,0,0,0,0,0]
# df = pd.DataFrame({
#     "text":texts,
#     "label":labels
# })
# print(df)
# print("-"*50)
# print(df["label"].value_counts())
# print("-"*50)

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
vectorizer = TfidfVectorizer(ngram_range=(1,2))
x = vectorizer.fit_transform(df["text"])
print(x.shape)
print(vectorizer.get_feature_names_out())
print("-"*50)

#建模
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,df["label"],test_size = 0.25,random_state=42
)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# print("预测：",y_pred)
# print("真实：",y_test.values)
# print("-"*50)

#经典机器学习文本分类 pipeline:文本 → TF-IDF → 逻辑回归 → 分类

# =========================
# 6️⃣ 查看模型学到的关键词权重
# coef 越小：越偏向 negative
# coef 越大：越偏向 positive
# 用来理解模型判断情感时更看重哪些词
# =========================
feature_names = vectorizer.get_feature_names_out()
cofficients = model.coef_[0]
word_coef = list(zip(feature_names,cofficients))
word_coef_sorted = sorted(word_coef,key=lambda x:x[1])
for word,coef in word_coef_sorted[:20]:
    print(word,round(coef,3))
print("-"*50)
for word,coef in word_coef_sorted[-20:]:
    print(word,round(coef,3))
print("-"*50)

#我们在输出时发现，模型会学到“无意义但有相关性”的特征，比如and 4.974    it 3.418    today 3.671

# =========================
# 预测新文本
# =========================
new_text = ["I love this movie"]
x_new = vectorizer.transform(new_text)
pred = model.predict(x_new)
print(pred)

# from sklearn.metrics import classification_report
# print(classification_report(y_test,y_pred))