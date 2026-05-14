import pandas as pd
df = pd.read_csv("train.csv")
print(df.shape)
print(df.head())
print(df.columns)
print("-"*50)

df.columns = ["label","title","description"]
df["text"] = df["title"] + " " + df["description"]
print(df[["label","text"]].head())
print("-"*50)

# =========================
# 构造词表(原始文本->split  构建词表)
# =========================
word_to_index = {}
index = 0 
for text in df["text"]:
    words = text.lower().split() #lower()全部转小写；split()按空格分词
    for word in words:
        if word not in word_to_index:
            word_to_index[word] = index
            index += 1
print("词表大小：",len(word_to_index))
print(list(word_to_index.items())[:10])
print("-"*50)

# =========================
# 文本转 token ids （查词表word_to_index  文本转token_ids）
# =========================
def text_to_ids(text):
    words = text.lower().split()
    ids = []
    for word in words:
        if word in word_to_index:
            ids.append(word_to_index[word])
    return ids
sample_text = df["text"].iloc[0]
sample_ids = text_to_ids(sample_text)
print("原始文本：")
print(sample_text)
print("转换后的token_ids：")
print(sample_ids)
print("长度：",len(sample_ids))
print("-"*50)

# =========================
# padding：统一句子长度(长度30)
# =========================
max_len = 30
def pad_sequence(ids,max_len):
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0]*(max_len - len(ids))
    return ids

sample_padded =  pad_sequence(sample_ids,max_len)
print("padding后:",sample_padded)
print("padding后长度：",len(sample_padded))
print("-"*50)

# =========================
# 处理全部文本
# =========================
all_ids = []
for text in df["text"]:
    ids = text_to_ids(text)
    padded_ids = pad_sequence(ids,max_len)
    all_ids.append(padded_ids)
print("样本数量：",len(all_ids))
print("第一条样本：",all_ids[0])
print("第一条样本长度：",len(all_ids[0]))
print("-"*50)

# =========================
# 转成 PyTorch Tensor
# =========================
import torch
x = torch.tensor(all_ids,dtype = torch.long) #把普通list转成tensor张量（tensorflow、pytorch更喜欢处理的形式）
y = torch.tensor(df["label"].values - 1,dtype = torch.long)
print("x shape:",x.shape)
print("y shape:",y.shape)
print("第一条x:",x[0])
print("第一条y:",y[0])
print("-"*50)

# =========================
# Embedding 层
# =========================
import torch.nn as nn
vocab_size = len(word_to_index)
embedding = nn.Embedding(
    num_embeddings = vocab_size,
    embedding_dim = 64 #向量维数
)
sample_embedding = embedding(x[:2])
print("输入shape:",x[:2].shape)
print("Embedding后的shape:",sample_embedding.shape)
print("-"*50)

# =========================
# LSTM 层
# =========================
lstm = nn.LSTM(
    input_size = 64,
    hidden_size = 128,
    batch_first = True
)
lstm_out,(h_n,c_n) = lstm(sample_embedding)
print("LSTM输出shape:",lstm_out.shape)
print("最后隐藏状态h_n.shape:",h_n.shape)
print("最后记忆状态c_n.shape:",c_n.shape)
print("-"*50)

# =========================
# 分类层
# =========================
classifier = nn.Linear(
    in_features = 128,
    out_features = 4
)
last_hidden = h_n[-1]
logits = classifier(last_hidden)
print("last_hidden.shape:",last_hidden.shape)
print("logits.shape:",logits.shape)
print("logits:",logits)
print("-"*50)

# =========================
# softmax 概率
# =========================
import torch.nn.functional as F
probs = F.softmax(logits,dim=1)
print("概率shape:",probs.shape)
print(probs)
print("-"*50)

# =========================
# Loss 损失函数
# =========================
criterion = nn.CrossEntropyLoss() #交叉熵损失函数，专门用于多分类任务
sample_y = y[:2]
loss = criterion(logits,sample_y) #这里传的是logits，不是softmax后的probs，因为CrossEntropyLoss会自动处理logits将其转换为softmax概率向量
print("真实标签sample_y:",sample_y)
print("loss:",loss.item())
print("-"*50)

# =========================
# 定义完整 LSTM 分类模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_size,num_classes)
    # 自己造一个LSTM文本分类模型，内有embedding层 + lstm层 + fc（分类层）
    def forward(self,x):
        embedded = self.embedding(x)
        lstm_out,(h_n,c_n) = self.lstm(embedded)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits
    # 数据前向传播的流程

# =========================
# 创建模型
# =========================
model = LSTMClassifier(
    vocab_size = vocab_size, #词表大学
    embedding_dim = 64, #每个词转化成64维向量
    hidden_size = 128, #隐藏层将一句话压缩成128维来理解学习
    num_classes = 4    #做4分类
)

# =========================
# 测试模型
# =========================
test_logits = model(x[:2])
print("test_logits.shape:",test_logits.shape)
print(test_logits)
print("-"*50)

# =========================
# optimizer 优化器
# =========================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)
print(optimizer)
print("-"*50)

# =========================
# 一次前向传播
# =========================
logits = model(x[:32]) #让模型处理32条新闻
batch_y = y[:32]
loss = criterion(logits,batch_y)
print("loss:",loss.item())
print("-"*50)

# =========================
# 反向传播
# =========================
optimizer.zero_grad() #清空旧梯度，每次反向传播前都要清空，不然梯度会累加
loss.backward() #根据loss反向计算每个参数要怎么修改才能降低损失
optimizer.step() # 修改参数
print("参数更新完成")
print("-"*50)

# =========================
# 小循环训练
# =========================
for epoch in range(5):
    #训练
    logits = model(x[:32])
    batch_y = y[:32]
    loss = criterion(logits,batch_y)
    #反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #打印/对比
    print("eopch:",epoch+1,"loss",loss.item())
preds = torch.argmax(logits,dim=1)
print("预测类别:", preds)
print("真实类别:", batch_y)

accuracy = (preds == batch_y).float().mean()
print("accuracy:",accuracy.item())

# =========================
# 划分训练集和测试集
# =========================
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state=42,stratify = y
)

# =========================
# 创建 DataLoader
# =========================
from torch.utils.data import TensorDataset,DataLoader
train_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test,y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = True
)
for batch_x,batch_y in train_loader:
    print("batch_x_shape",batch_x.shape)
    print("batch_y_shape",batch_y.shape)
    break


# =========================
# 正式训练
# =========================
for epoch in range(3):
    total_loss = 0
    for batch_x,batch_y in train_loader:
        logits = model(batch_x)
        loss = criterion(logits,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("epoch:",epoch+1)
    print("total_loss:",total_loss)
    print("-"*50)

# =========================
# 测试集评估
# =========================
corret = 0
total = 0
with torch.no_grad():
    for batch_x,batch_y in test_loader:
        logits = model(batch_x)
        preds = torch.argmax(logits,dim=1)
        corret += (preds == batch_y).sum().item()
        total += batch_y.size(0)
accuracy = corret/total
print("test_accuracy:",accuracy)











