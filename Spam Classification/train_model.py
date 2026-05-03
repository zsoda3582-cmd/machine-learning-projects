import pandas as pd


import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]"," ",text)
    return text

df = pd.read_csv("spam.csv",encoding = "latin-1")
df = df[["v1","v2"]]
df.columns = ["label","text"]
print(df.head)
print(df["label"].value_counts())
print("-"*50)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df["text"] = df["text"].apply(clean_text)
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["text"])
y = df["label"].map({"ham":0,"spam":1})
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state = 42,stratify = y
)
model = LogisticRegression()
model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
for t in [0.7,0.5,0.4,0.3,0.2,0.1]:
    y_prob = model.predict_proba(x_test)[:,1]
    y_pred = (y_prob > t).astype(int)
    print("threshold:",t,",\nclassification_report:\n",classification_report(y_test,y_pred))

import matplotlib.pyplot as plt
thresholds = [0.7,0.5,0.4,0.3,0.2,0.1]
recalls = [0.54,0.73,0.81,0.85,0.91,0.97]
precisions = [0.99,0.99,0.98,0.96,0.94,0.67]
plt.plot(thresholds,recalls,label="recall")
plt.plot(thresholds,precisions,label="precision")
plt.gca().invert_xaxis()
plt.legend()
plt.grid()
plt.show()