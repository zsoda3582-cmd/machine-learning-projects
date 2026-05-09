import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")
print(df.head())
print(df.columns)
print(df["sentiment"].value_counts())
df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})
print(df["sentiment"].value_counts())
print("-"*50)
texts = df["review"]
labels = df["sentiment"]
x = texts
y = labels
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state = 42,stratify = y
)
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_train_pad = pad_sequences(x_train_seq,maxlen=200)
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_test_pad = pad_sequences(x_test_seq,maxlen=200)
print(x_train_seq[0])
print(x_train_pad[0])
print(x_train_pad.shape)
print(x_test_pad.shape)

model = Sequential() #Sequential()本质上是空模型，现在开始搭模型了
model.add(Embedding(input_dim=10000,output_dim=64)) #model.add()——开始往神经网络里加层，此处加embedding层
model.add(LSTM(64)) #加LSTM层
model.add(Dense(1,activation = "sigmoid")) #加Dense层（输出层）
model.compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)
history = model.fit(
    x_train_pad,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)
loss,accuracy = model.evaluate(x_test_pad,y_test)
print("text accuracy:",accuracy)

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq,maxlen=200)
    prob = model.predict(pad)[0][0]
    if prob >= 0.5:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment,prob

while True:
    text = input("请输入文本（exit表示退出)：")
    if text == "exit":
        break
    sentiment,prob = predict_sentiment(text)
    print("预测情感：",sentiment)
    print("positive概率：",round(prob,4))



"""
tokenizer 将文字转换成数字向量
seq 处理序列
pad 将向量长度补齐（此处设置的是200）
Embedding 数字id变成向量
LSTM 处理文本向量
Dense(sigmoid) 输出正面/负面概率
Tensorflow中有一整套完整的训练系统，fit()负责训练，evaluate()负责测试，predict()负责预测，都已经封装好了
"""