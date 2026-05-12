import torch
import torch.nn as nn
sentence = "I love this movie"
words = sentence.split()
print(words)

# =========================
# 给每个词编号
# =========================
word_to_index = {}
for i,word in enumerate(words):
    word_to_index[word] = i
print(word_to_index)

# =========================
# 生成 skip-gram 训练数据
# =========================
window_size = 1
pairs = []
for i,word in enumerate(words):
    center_word = word
    #左边词
    if i-1 >= 0:
        context_word = words[i-1]
        pairs.append((center_word,context_word))
    #右边词
    if i+1 < len(words):
        context_word = words[i+1]
        pairs.append((center_word,context_word))
print(pairs)

# =========================
# pair列表转成数字 训练数据
# =========================
input_data = []
target_data = []
for center,context in pairs:
    input_data.append(word_to_index[center])
    target_data.append(word_to_index[context_word])
print(input_data)
print(target_data)

# =========================
# 创建 Embedding 层
# =========================
embedding = nn.Embedding(
    num_embeddings = 4,#4个词
    embedding_dim = 5  #5维向量
)
print(embedding.weight)

input_word = torch.tensor([1])
vector = embedding(input_word)
print(vector)

# =========================
# 输入一句话
# =========================
sentence_ids = torch.tensor([0,1,2,3])
sentence_vectors = embedding(sentence_ids)
print(sentence_vectors)
print(sentence_vectors.shape)

# =========================
# 建模
# =========================
from gensim.models import Word2Vec

# =========================
# 准备训练数据
# =========================
sentences = [
    ["i", "love", "this", "movie"],
    ["i", "like", "this", "film"],
    ["this", "movie", "is", "great"],
    ["this", "film", "is", "good"],
    ["the", "movie", "was", "amazing"],
    ["the", "film", "was", "excellent"],
    ["i", "watched", "this", "movie"],
    ["i", "watched", "this", "film"],
    ["movie", "and", "film", "are", "similar"]
]

# =========================
# 训练 Word2Vec
# =========================
model = Word2Vec(
    sentences,
    vector_size = 20,
    window = 2,
    min_count = 1,
    sg = 1,
    epochs = 1000
)

# =========================
# 查看词向量
# =========================
print(model.wv["movie"])
print(model.wv["film"])
print(model.wv.similarity("movie","film"))
print(model.wv.most_similar("movie"))