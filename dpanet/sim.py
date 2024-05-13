import torch
import torch.nn as nn

# 创建一个嵌入层，其中num_embeddings=10表示嵌入层可以容纳10个类别（例如，10个不同的词），
# embedding_dim=3表示每个类别将被嵌入到一个3维的向量中。
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# 模拟输入数据：一批大小为2的索引序列，每个序列长度为4。
X = torch.LongTensor([[1,2,4,5],[4,3,2,9]])

print(X.shape)
# 通过嵌入层获取嵌入结果
output = embedding(X)

print(output.shape)


embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)

# 模拟输入数据，其中0被用作填充值。
X = torch.LongTensor([[0,2,0,5],[4,0,2,0]])

# 通过嵌入层获取嵌入结果
output = embedding(X)

print(output)



# 假设我们有一些预训练的嵌入向量
pretrained_embeddings = torch.FloatTensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])

# 创建嵌入层，其大小与预训练的嵌入向量相匹配
embedding = nn.Embedding(num_embeddings=4, embedding_dim=3)

# 使用预训练的嵌入向量初始化嵌入层
embedding.weight = nn.Parameter(pretrained_embeddings)

# 模拟输入数据
X = torch.LongTensor([[0,1],[2,3]])

# 通过嵌入层获取嵌入结果
output = embedding(X)

print(output)


# 创建一个嵌入层，其中max_norm=1表示嵌入向量的最大范数被限制为1。
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, max_norm=1)

# 模拟输入数据
X = torch.LongTensor([[1,2,4,5],[4,3,2,9]])

# 通过嵌入层获取嵌入结果
output = embedding(X)

print(output)