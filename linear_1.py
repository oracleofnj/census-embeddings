import numpy as np
import torch
from torch import nn, optim

np.random.seed(42)
x = np.random.rand(50)
y_train_1 = torch.Tensor(2 * x + 0.5).view(-1, 1)
y_train_2 = torch.Tensor((-1) * x).view(-1, 1)


class LinearRegressionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class EmbeddingModule(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, x):
        out = self.embeddings(x)
        return out


class LinearEmbeddingModule(nn.Module):
    def __init__(self, embeddings, output_dim):
        super().__init__()
        self.embeddings = embeddings
        self.linear = LinearRegressionModule(
            embeddings.embedding_dim,
            output_dim
        )

    def forward(self, x):
        embeddings = self.embeddings(x)
        out = self.linear(embeddings)
        return out


embeddings = EmbeddingModule(50, 1)
model_1 = LinearEmbeddingModule(embeddings, 1)
model_2 = LinearEmbeddingModule(embeddings, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD([
    {'params': model_1.linear.parameters()},
    {'params': model_2.linear.parameters()},
    {'params': model_1.embeddings.parameters(), 'lr': 0.5},
], lr=0.1)

indices = torch.LongTensor([x for x in range(50)])

for epoch in range(2000):
    optimizer.zero_grad()
    outputs_1 = model_1.forward(indices)
    loss_1 = criterion(outputs_1, y_train_1)
    loss_1.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs_2 = model_2.forward(indices)
    loss_2 = criterion(outputs_2, y_train_2)
    loss_2.backward()
    optimizer.step()

    if ((epoch + 1) % 100) == 0:
        print('epoch {0}, loss 1 {1}, loss 2 {2}'.format(
            epoch,
            loss_1,
            loss_2
        ))

print(model_1.linear.state_dict())
print(model_2.linear.state_dict())
print(np.corrcoef(x, embeddings.embeddings.weight.data, rowvar=False))
