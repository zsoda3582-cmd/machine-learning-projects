import pandas as pd
df = pd.read_csv("train.csv")
print(df.head())
print(df.columns)
print(df["Class Index"].value_counts())
print("-"*50)

df.columns = ["label","title","description"]
df["text"] = df["title"] + " " + df["description"]
print(df[["label","text"]].head())

#导包
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["text"])
y = df["label"]

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state = 42,stratify = y
)
model = LogisticRegression(max_iter = 1000)
model_nb = MultinomialNB()
model_svc = LinearSVC()
model.fit(x_train,y_train)
model_nb.fit(x_train,y_train)
model_svc.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_nb = model_nb.predict(x_test)
y_pred_svc = model_svc.predict(x_test)
print("classifiction_report:\n",classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("confusion_matrix:\n",cm)
print("classifiction_report_nb:\n",classification_report(y_test,y_pred_nb))
cm_nb = confusion_matrix(y_test,y_pred_nb)
print("confusion_matrix_nb:\n",cm_nb)
print("classifiction_report_svc:\n",classification_report(y_test,y_pred_svc))
cm_svc = confusion_matrix(y_test,y_pred_svc)
print("confusion_matrix_svc:\n",cm_svc)
# label_map = {
#     1:"world",
#     2:"sport",
#     3:"business",
#     4:"sci/tech"
# }
# feature_names = vectorizer.get_feature_names_out()
# for i,label_name in label_map.items():
#     coefs = model.coef_[i-1]
#     top10 = coefs.argsort()[-10:][::-1]
#     print("\n类别：",label_name)
#     print("最重要的词：")
#     for idx in top10:
#         print(feature_names[idx],round(coefs[idx],2))


# def predict_news(text):
#     x_new = vectorizer.transform([text])
#     prob = model.predict_proba(x_new)[0]
#     pred = model.predict(x_new)[0]
#     top_2 = prob.argsort()[-2:][::-1]
#     return pred,prob,top_2
# while True:
#     text = input("输入一句新闻（输入exit退出）：")
#     if text =="exit":
#         break
#     pred,prob,top_2 = predict_news(text)
#     print("预测类别：",label_map[pred])
#     print("Top2概率:")
#     for idx in top_2:
#         label = idx + 1
#         print(label_map[label],round(prob[idx],2))

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot = True,
    fmt = "d",
    cmap = "Blues"
)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Confusion_Matrix")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_nb,
    annot = True,
    fmt = "d",
    cmap = "Blues"
)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Confusion_Matrix_nb")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_svc,
    annot = True,
    fmt = "d",
    cmap = "Blues"
)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Confusion_Matrix_svc")
plt.show()